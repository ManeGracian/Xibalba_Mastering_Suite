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

    Meter RT-safe para uso en plugins (Mix/Mastering):
    - Peak (por bloque) + ballistics (attack/release)
    - RMS (por bloque) + ballistics
    - Thread-safe: audio thread publica valores; GUI thread los lee vía atomics
    - Channel-agnostic: 1..N canales

    Notas:
    - "True Peak" aquí es peak del bloque procesado. Si el bloque viene
      oversampled (por ejemplo después de un oversampler), se aproxima
      a true peak inter-sample.
    - No asigna memoria ni bloquea en process().
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
        updateBallisticsIfNeeded();

        // 2) Peak del bloque (máximo abs entre canales)
        SampleType blockPeak = SampleType (0);

        for (int ch = 0; ch < numCh; ++ch)
        {
            auto* data = block.getChannelPointer ((size_t) ch);
            auto mm = juce::FloatVectorOperations::findMinAndMax (data, numSamples);

            const auto absMax = (SampleType) juce::jmax (std::abs (mm.getStart()),
                                                        std::abs (mm.getEnd()));

            if (absMax > blockPeak)
                blockPeak = absMax;
        }

        // 3) RMS del bloque (energía promedio entre canales)
        double sumSquares = 0.0;
        const double denom = (double) numCh * (double) numSamples;

        for (int ch = 0; ch < numCh; ++ch)
        {
            auto* data = block.getChannelPointer ((size_t) ch);

            for (int i = 0; i < numSamples; ++i)
            {
                const double x = (double) data[i];
                sumSquares += x * x;
            }
        }

        const SampleType blockRms = (SampleType) std::sqrt (sumSquares / juce::jmax (denom, 1.0));

        // 4) Ballistics (attack/release)
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

        // Defaults de lectura agradable (tú puedes ajustar)
        attackMsTarget.store  (5.0,   std::memory_order_release);
        releaseMsTarget.store (300.0, std::memory_order_release);
        coeffsDirty.store (true, std::memory_order_release);

        updateBallisticsIfNeeded (true);
        reset();
    }

private:
    //==========================================================================
    void updateBallisticsIfNeeded (bool force = false) noexcept
    {
        if (!force && !coeffsDirty.exchange (false, std::memory_order_acq_rel))
            return;

        if (sampleRate <= 0.0)
            return;

        const double atkMs = attackMsTarget.load (std::memory_order_acquire);
        const double relMs = releaseMsTarget.load (std::memory_order_acquire);

        // Coeficientes exponenciales por muestra:
        // y += a*(x - y), a = 1 - exp(-1/(T*srate))
        const double atkSamps = juce::jmax (0.0001, atkMs * 0.001) * sampleRate;
        const double relSamps = juce::jmax (0.0001, relMs * 0.001) * sampleRate;

        attackCoeff  = (SampleType) (1.0 - std::exp (-1.0 / atkSamps));
        releaseCoeff = (SampleType) (1.0 - std::exp (-1.0 / relSamps));
    }

    SampleType applyBallistics (SampleType current, SampleType target) const noexcept
    {
        const auto c = (target > current) ? attackCoeff : releaseCoeff;
        return current + c * (target - current);
    }

    //==========================================================================
    // Estado (audio thread)
    double sampleRate { 44100.0 };

    SampleType peakBallistic { SampleType (0) };
    SampleType rmsBallistic  { SampleType (0) };

    SampleType attackCoeff  { SampleType (0.01) };
    SampleType releaseCoeff { SampleType (0.001) };

    // Publicación (audio -> GUI)
    std::atomic<float> peakLin { 0.0f };
    std::atomic<float> rmsLin  { 0.0f };

    // Targets (GUI -> audio)
    std::atomic<double> attackMsTarget  { 5.0 };
    std::atomic<double> releaseMsTarget { 300.0 };
    std::atomic<bool> coeffsDirty { true };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Meter)
};

} // namespace xb

