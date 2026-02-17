#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cmath>

#include "XC_DSPBlockBase.h"

namespace xb
{

/*
==============================================================================
    xb::Meter<SampleType>

    Meter RT-safe para plugins (Mix/Mastering):
    - Peak (por bloque) + ballistics (attack/release)
    - RMS  (por bloque) + ballistics
    - Thread-safe: audio thread publica valores; GUI thread los lee vía atomics
    - Channel-agnostic: 1..N canales

    Ballistics (FIX importante):
    - Este meter mide Peak/RMS "por bloque" y aplica ballistics UNA vez por bloque.
      Para que attack/release se mantengan consistentes aunque cambie el block size,
      los coeficientes se calculan con dt = duración del bloque (segundos),
      NO por muestra.

    Compatibilidad:
    - No usa FloatVectorOperations::computeRMS ni ::dotProduct (pueden no existir).
    - RMS se calcula con loop simple (seguro y portable).
    - Peak usa findMinAndMax (esto sí suele estar en JUCE desde hace años).
==============================================================================
*/

template <typename SampleType>
class Meter final : public DSPBlockBase
{
public:
    Meter() = default;
    ~Meter() override = default;

    //==========================================================================
    void reset() noexcept override
    {
        peakBallistic = SampleType (0);
        rmsBallistic  = SampleType (0);

        peakLin.store (0.0f, std::memory_order_release);
        rmsLin.store  (0.0f, std::memory_order_release);
    }

    //==========================================================================
    // Thread-safe setters (GUI/host thread -> audio thread)
    void setAttackMs (SampleType ms) noexcept
    {
        ms = juce::jlimit (SampleType (0.1), SampleType (500.0), ms);
        attackMsTarget.store ((double) ms, std::memory_order_release);
        coeffsDirty.store (true, std::memory_order_release);
    }

    void setReleaseMs (SampleType ms) noexcept
    {
        ms = juce::jlimit (SampleType (1.0), SampleType (5000.0), ms);
        releaseMsTarget.store ((double) ms, std::memory_order_release);
        coeffsDirty.store (true, std::memory_order_release);
    }

    //==========================================================================
    // Procesamiento (AudioBuffer)
    void process (const juce::AudioBuffer<SampleType>& buffer) noexcept
    {
        const juce::dsp::AudioBlock<const SampleType> block (buffer);
        process (block);
    }

    // Procesamiento (AudioBlock) - recomendado para chain/grafo
    void process (const juce::dsp::AudioBlock<const SampleType>& block) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
            return;

        const int numCh      = (int) block.getNumChannels();
        const int numSamples = (int) block.getNumSamples();

        if (numCh <= 0 || numSamples <= 0)
            return;

        // 1) Actualizar coeficientes si hubo cambios (audio thread)
        updateBallisticsIfNeeded (numSamples);

        // 2) Peak del bloque (máximo abs entre canales)
        SampleType blockPeak = SampleType (0);

        for (int ch = 0; ch < numCh; ++ch)
        {
            const auto* data = block.getChannelPointer ((size_t) ch);
            const auto mm = juce::FloatVectorOperations::findMinAndMax (data, numSamples);

            const auto absMax = (SampleType) juce::jmax (std::abs (mm.getStart()),
                                                         std::abs (mm.getEnd()));

            if (absMax > blockPeak)
                blockPeak = absMax;
        }

        // 3) RMS global del bloque (energía promedio entre canales)
        //    RMS_global = sqrt( mean(x^2) ) sobre todos los samples y canales.
        double sumSquares = 0.0;

        for (int ch = 0; ch < numCh; ++ch)
        {
            const auto* data = block.getChannelPointer ((size_t) ch);

            // Loop portable
            for (int i = 0; i < numSamples; ++i)
            {
                const double x = (double) data[i];
                sumSquares += x * x;
            }
        }

        const double denom = juce::jmax (1.0, (double) numCh * (double) numSamples);
        const SampleType blockRms = (SampleType) std::sqrt (sumSquares / denom);

        // 4) Ballistics (por bloque)
        peakBallistic = applyBallistics (peakBallistic, blockPeak);
        rmsBallistic  = applyBallistics (rmsBallistic,  blockRms);

        // 5) Publicar (audio -> GUI)
        peakLin.store ((float) peakBallistic, std::memory_order_release);
        rmsLin.store  ((float) rmsBallistic,  std::memory_order_release);
    }

    //==========================================================================
    // Getters thread-safe (GUI thread)
    [[nodiscard]] float getPeakLinear() const noexcept
    {
        return peakLin.load (std::memory_order_acquire);
    }

    [[nodiscard]] float getRmsLinear() const noexcept
    {
        return rmsLin.load (std::memory_order_acquire);
    }

    [[nodiscard]] float getPeakDb() const noexcept
    {
        return juce::Decibels::gainToDecibels (getPeakLinear(), -120.0f);
    }

    [[nodiscard]] float getRmsDb() const noexcept
    {
        return juce::Decibels::gainToDecibels (getRmsLinear(), -120.0f);
    }

protected:
    //==========================================================================
    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        if (spec.sampleRate <= 0.0)
            return;

        sampleRate = spec.sampleRate;

        // Defaults (ajustables)
        attackMsTarget.store  (5.0,   std::memory_order_release);
        releaseMsTarget.store (300.0, std::memory_order_release);
        coeffsDirty.store (true, std::memory_order_release);

        lastBlockSize.store (0, std::memory_order_release);
        reset();
    }

private:
    //==========================================================================
    // Coeficientes POR BLOQUE: a = 1 - exp(-dt / T)
    void updateBallisticsIfNeeded (int currentBlockSize) noexcept
    {
        const int prev = lastBlockSize.load (std::memory_order_acquire);

        const bool blockSizeChanged = (currentBlockSize != prev);
        const bool dirty = coeffsDirty.exchange (false, std::memory_order_acq_rel);

        if (!dirty && !blockSizeChanged)
            return;

        if (sampleRate <= 0.0 || currentBlockSize <= 0)
            return;

        lastBlockSize.store (currentBlockSize, std::memory_order_release);

        const double atkMs = attackMsTarget.load (std::memory_order_acquire);
        const double relMs = releaseMsTarget.load (std::memory_order_acquire);

        const double dt = (double) currentBlockSize / sampleRate;  // segundos por bloque

        const double atkT = juce::jmax (0.0001, atkMs * 0.001);
        const double relT = juce::jmax (0.0001, relMs * 0.001);

        attackCoeff  = (SampleType) (1.0 - std::exp (-dt / atkT));
        releaseCoeff = (SampleType) (1.0 - std::exp (-dt / relT));

        attackCoeff  = juce::jlimit (SampleType (0), SampleType (1), attackCoeff);
        releaseCoeff = juce::jlimit (SampleType (0), SampleType (1), releaseCoeff);
    }

    SampleType applyBallistics (SampleType current, SampleType target) const noexcept
    {
        const auto c = (target > current) ? attackCoeff : releaseCoeff;
        return current + c * (target - current);
    }

    //==========================================================================
    double sampleRate { 44100.0 };

    SampleType peakBallistic { SampleType (0) };
    SampleType rmsBallistic  { SampleType (0) };

    SampleType attackCoeff  { SampleType (0.5) };
    SampleType releaseCoeff { SampleType (0.1) };

    // Publicación (audio -> GUI)
    std::atomic<float> peakLin { 0.0f };
    std::atomic<float> rmsLin  { 0.0f };

    // Targets (GUI -> audio)
    std::atomic<double> attackMsTarget  { 5.0 };
    std::atomic<double> releaseMsTarget { 300.0 };
    std::atomic<bool> coeffsDirty { true };

    // Para recalcular coeficientes si cambia block size
    std::atomic<int> lastBlockSize { 0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Meter)
};

} // namespace xb

