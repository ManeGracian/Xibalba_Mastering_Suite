#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <vector>
#include <cmath>
#include "XC_DSPBlockBase.h"

namespace xb
{

/*
==============================================================================
    XC_DynamicsCore<SampleType>

    Core base para procesadores dinámicos (Comp, Limiter, etc.)
    Responsabilidades:
    - Lookahead (delay circular con interpolación)
    - Sidechain HPF simple (one-pole) por canal
    - Detector peak + ballistics (attack/release)
    - Hook virtual: calculateGainLinear(detectorLin)

    Diseño RT-safe:
    - NO alloc/resize/locks en process()
    - Los setters son thread-safe: escriben atomics (targets)
    - El audio thread “consume” targets al inicio de cada bloque

    Channel-agnostic:
    - Estados y delay buffer se dimensionan por spec.numChannels en prepare()
    - En process: se procesa min(numChActual, numChPreparados)

==============================================================================
*/

template <typename SampleType>
class XC_DynamicsCore : public DSPBlockBase
{
public:
    XC_DynamicsCore() = default;
    ~XC_DynamicsCore() override = default;

    //==========================================================================
    void reset() noexcept override
    {
        delayBuffer.clear();
        std::fill (scFilterStates.begin(), scFilterStates.end(), SampleType (0));

        envelopeState    = SampleType (0);
        currentGainState = SampleType (1);
        writePos         = 0;

        lookAheadSmoother.setCurrentAndTargetValue (SampleType (0));
    }

    //==========================================================================
    // Thread-safe SETTERS (UI/host thread)
    void setThreshold (SampleType db) noexcept
    {
        thresholdDbTarget.store ((double) db, std::memory_order_release);
    }

    void setRatio (SampleType r) noexcept
    {
        const auto safe = (double) juce::jmax (SampleType (1.0), r);
        ratioTarget.store (safe, std::memory_order_release);
    }

    void setKnee (SampleType db) noexcept
    {
        kneeWidthTarget.store ((double) juce::jmax (SampleType (0.0), db), std::memory_order_release);
    }

    void setMakeupGain (SampleType db) noexcept
    {
        makeupDbTarget.store ((double) db, std::memory_order_release);
    }

    void setAttack (SampleType ms) noexcept
    {
        attackMsTarget.store ((double) juce::jmax (SampleType (0.1), ms), std::memory_order_release);
    }

    void setRelease (SampleType ms) noexcept
    {
        releaseMsTarget.store ((double) juce::jmax (SampleType (0.1), ms), std::memory_order_release);
    }

    void setLookAhead (SampleType ms) noexcept
    {
        // Tu intención original era 0..20ms
        const auto clamped = (double) juce::jlimit (SampleType (0), SampleType (maxLookAheadMs), ms);
        lookAheadMsTarget.store (clamped, std::memory_order_release);
    }

    void setSidechainHPF (SampleType freqHz) noexcept
    {
        scHpfHzTarget.store ((double) juce::jmax (SampleType (0.0), freqHz), std::memory_order_release);
    }

    //==========================================================================
    // GETTERS (para meters/GUI)
    [[nodiscard]] SampleType getCurrentGain() const noexcept { return currentGainState; }

    // Latencia instantánea (samples) del lookahead actual (suavizado)
    [[nodiscard]] int getLatencySamples() const noexcept override
    {
        return (int) lookAheadSmoother.getCurrentValue();
    }

    //==========================================================================
    // Procesamiento (AudioBuffer)
    void process (juce::AudioBuffer<SampleType>& buffer) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
            return;

        const int numSamples = buffer.getNumSamples();
        const int inCh       = buffer.getNumChannels();

        if (numSamples <= 0 || inCh <= 0)
            return;

        const int delaySize = delayBuffer.getNumSamples();
        if (delaySize <= 1)
            return;

        // 1) Consumir targets -> actualizar coeficientes/smoothers (audio thread)
        updateFromTargetsIfNeeded();

        // 2) Procesar solo canales para los que tenemos estado/buffer
        const int preparedCh = (int) scFilterStates.size();
        const int chToProcess = juce::jmin (inCh, preparedCh);

        auto* const* inputs  = buffer.getArrayOfReadPointers();
        auto** outputs       = buffer.getArrayOfWritePointers();
        auto** delayW        = delayBuffer.getArrayOfWritePointers();
        auto* const* delayR  = delayBuffer.getArrayOfReadPointers();

        // Ganancia smoothing ligera (evita stepping). Esto es “taste”.
        constexpr SampleType gainSmoothing = SampleType (0.005);

        for (int i = 0; i < numSamples; ++i)
        {
            // Lookahead en samples (puede ser fraccional por smoothing)
            const SampleType currentDelay = lookAheadSmoother.getNextValue();

            // ------------------------------------------------------------
            // STEP 1: Write a delay buffer + detector (maxAbs over channels)
            SampleType maxAbs = SampleType (0);

            for (int ch = 0; ch < chToProcess; ++ch)
            {
                const SampleType x = inputs[ch][i];

                // Escribir audio original al delay
                delayW[ch][writePos] = x;

                // Sidechain HPF one-pole (simple y barato)
                // state += a*(x - state);  hp = x - state;
                scFilterStates[(size_t) ch] += scHpfCoeff * (x - scFilterStates[(size_t) ch]);
                const SampleType detectorSignal = x - scFilterStates[(size_t) ch];

                const SampleType s = std::abs (detectorSignal);
                if (s > maxAbs) maxAbs = s;
            }

            // Si hay canales extra sin estado preparado, los pasamos tal cual a delay (opcional).
            // Aquí los copiamos para que no queden “vacíos” en salida si el host entrega más canales.
            for (int ch = chToProcess; ch < inCh; ++ch)
            {
                const SampleType x = inputs[ch][i];
                // Si delayBuffer no tiene ese canal, no podemos escribirlo; simplemente seguirá ruta directa.
                outputs[ch][i] = x; // passthrough (conservador)
            }

            // ------------------------------------------------------------
            // STEP 2: Ballistics (attack/release)
            if (maxAbs > envelopeState)
                envelopeState += attackCoeff  * (maxAbs - envelopeState);
            else
                envelopeState += releaseCoeff * (maxAbs - envelopeState);

            // ------------------------------------------------------------
            // STEP 3: Ganancia (hook del hijo)
            const SampleType detectorLin = juce::jmax (envelopeState, SampleType (1e-12));
            const SampleType rawGain     = calculateGainLinear (detectorLin);

            currentGainState += gainSmoothing * (rawGain - currentGainState);

            // ------------------------------------------------------------
            // STEP 4: Leer delay con interpolación lineal
            // readPos = writePos - delay (wrap)
            double readPos = (double) writePos - (double) currentDelay;
            if (readPos < 0.0)
                readPos += (double) delaySize;

            const int idxA = (int) readPos;                  // floor
            const int idxB = (idxA + 1) % delaySize;
            const SampleType frac = (SampleType) (readPos - (double) idxA);

            // Aplicar ganancia + makeup a la señal delay
            for (int ch = 0; ch < chToProcess; ++ch)
            {
                const SampleType a = delayR[ch][idxA];
                const SampleType b = delayR[ch][idxB];
                const SampleType delayed = a + frac * (b - a);

                outputs[ch][i] = delayed * currentGainState * makeupGainLin;
            }

            // Avanzar write head
            writePos = (writePos + 1) % delaySize;
        }
    }

protected:
    //==========================================================================
    // Hook obligatorio: cada comp/limit implementa su curva
    virtual SampleType calculateGainLinear (SampleType detectorLin) const noexcept = 0;

    // Parámetros compartidos (snapshot en audio thread, disponibles para hijos si quieres)
    SampleType thresholdDb { SampleType (0) };
    SampleType ratio       { SampleType (1) };
    SampleType slope       { SampleType (0) };
    SampleType kneeWidth   { SampleType (0) };
    SampleType makeupGainLin { SampleType (1) };

    //==========================================================================
    void prepareInternal (const juce::dsp::ProcessSpec& specIn) override
    {
        if (specIn.sampleRate <= 0.0 || specIn.maximumBlockSize == 0 || specIn.numChannels == 0)
            return;

        // Lookahead máximo interno (20ms por tu clamp) + margen de block
        const int maxDelaySamples =
            (int) std::ceil (specIn.sampleRate * (maxLookAheadMs * 0.001)) +
            (int) specIn.maximumBlockSize + 2;

        delayBuffer.setSize ((int) specIn.numChannels, maxDelaySamples, false, false, true);
        delayBuffer.clear();

        scFilterStates.assign ((size_t) specIn.numChannels, SampleType (0));

        lookAheadSmoother.reset (specIn.sampleRate, 0.05);          // smoothing 50ms (tu intención original)
        lookAheadSmoother.setCurrentAndTargetValue (SampleType (0));

        writePos = 0;
        envelopeState = SampleType (0);
        currentGainState = SampleType (1);

        // Sincroniza coeficientes y snapshot iniciales desde targets
        updateFromTargetsIfNeeded (true);
        reset();
    }

private:
    //==========================================================================
    void updateFromTargetsIfNeeded (bool force = false) noexcept
    {
        const double sr = getSampleRate();
        if (sr <= 0.0)
            return;

        // Leer targets atomics
        const double thrDb = thresholdDbTarget.load (std::memory_order_acquire);
        const double rat   = ratioTarget.load      (std::memory_order_acquire);
        const double knee  = kneeWidthTarget.load  (std::memory_order_acquire);
        const double mkDb  = makeupDbTarget.load   (std::memory_order_acquire);
        const double atkMs = attackMsTarget.load   (std::memory_order_acquire);
        const double relMs = releaseMsTarget.load  (std::memory_order_acquire);
        const double laMs  = lookAheadMsTarget.load(std::memory_order_acquire);
        const double hpfHz = scHpfHzTarget.load    (std::memory_order_acquire);

        // Si nada cambió y no force, podrías evitar trabajo extra.
        // Aquí lo mantenemos simple (barato) y actualizamos siempre por seguridad.

        thresholdDb = (SampleType) thrDb;
        ratio       = (SampleType) juce::jmax (1.0, rat);
        slope       = (SampleType) (1.0 - (1.0 / (double) ratio));
        kneeWidth   = (SampleType) juce::jmax (0.0, knee);

        makeupGainLin = (SampleType) juce::Decibels::decibelsToGain ((SampleType) mkDb);

        // Coefs attack/release (T = ms)
        const double atkSamps = juce::jmax (0.0001, atkMs * 0.001) * sr;
        const double relSamps = juce::jmax (0.0001, relMs * 0.001) * sr;

        attackCoeff  = (SampleType) (1.0 - std::exp (-1.0 / atkSamps));
        releaseCoeff = (SampleType) (1.0 - std::exp (-1.0 / relSamps));

        // Sidechain HPF (one-pole) coef
        // a = 1 - exp(-2*pi*fc/sr)
        const double fc = juce::jlimit (0.0, sr * 0.49, hpfHz);
        scHpfCoeff = (SampleType) (1.0 - std::exp (-2.0 * juce::MathConstants<double>::pi * fc / sr));

        // Lookahead en samples (target)
        const SampleType laSamples = (SampleType) (juce::jlimit (0.0, (double) maxLookAheadMs, laMs) * 0.001 * sr);
        lookAheadSmoother.setTargetValue (laSamples);

        if (force)
            lookAheadSmoother.setCurrentAndTargetValue (laSamples);
    }

    //==========================================================================
    // CONSTANTES
    static constexpr double maxLookAheadMs = 20.0;

    //==========================================================================
    // Audio state
    juce::AudioBuffer<SampleType> delayBuffer;
    juce::SmoothedValue<SampleType, juce::ValueSmoothingTypes::Linear> lookAheadSmoother;

    std::vector<SampleType> scFilterStates;

    int writePos { 0 };
    SampleType envelopeState { SampleType (0) };
    SampleType currentGainState { SampleType (1) };

    // Coefs
    SampleType attackCoeff  { SampleType (0) };
    SampleType releaseCoeff { SampleType (0) };
    SampleType scHpfCoeff   { SampleType (0) };

    //==========================================================================
    // Thread-safe targets (setters write here)
    std::atomic<double> thresholdDbTarget { 0.0 };
    std::atomic<double> ratioTarget       { 1.0 };
    std::atomic<double> kneeWidthTarget   { 0.0 };
    std::atomic<double> makeupDbTarget    { 0.0 };

    std::atomic<double> attackMsTarget    { 10.0 };
    std::atomic<double> releaseMsTarget   { 100.0 };

    std::atomic<double> lookAheadMsTarget { 0.0 };
    std::atomic<double> scHpfHzTarget     { 20.0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_DynamicsCore)
};

} // namespace xb
