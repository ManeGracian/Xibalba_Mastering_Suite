#pragma once

#include <JuceHeader.h>
#include <atomic>
#include "XC_DSPBlockBase.h"

namespace xb
{

/*
==============================================================================
    xb::Gain<SampleType>

    Gain block (Mastering-grade) con smoothing multiplicativo.

    Objetivos:
    - RT-safe: sin allocs / locks en process()
    - Thread-safe: setters pueden llamarse desde UI/host thread (atomics)
    - Channel-agnostic: 1..N canales
    - Rutas rápidas: unity bypass, sin smoothing, y optimización de silencio

    Nota importante:
    - SmoothedValue NO es thread-safe si se modifica desde UI thread mientras el
      audio thread avanza getNextValue(). Por eso los setters escriben atomics,
      y el audio thread aplica cambios al inicio del bloque.
==============================================================================
*/

template <typename SampleType>
class Gain final : public DSPBlockBase
{
public:
    Gain() = default;
    ~Gain() override = default;

    //==========================================================================
    void reset() noexcept override
    {
        // RT-safe (sin allocs)
        const auto g = (SampleType) gainTargetLinear.load (std::memory_order_acquire);
        smoother.setCurrentAndTargetValue (juce::jmax (g, SampleType (0)));
    }

    //==========================================================================
    // Thread-safe setters (UI/host thread)
    void setGainLinear (SampleType newGain) noexcept
    {
        newGain = juce::jmax (newGain, SampleType (0));

        gainTargetLinear.store ((double) newGain, std::memory_order_release);
        gainDirty.store (true, std::memory_order_release);
    }

    void setGainDecibels (SampleType newGainDb) noexcept
    {
        setGainLinear (juce::Decibels::decibelsToGain (newGainDb, SampleType (-100.0)));
    }

    void setSmoothingTimeSeconds (SampleType timeSeconds) noexcept
    {
        timeSeconds = juce::jmax (timeSeconds, SampleType (0));

        smoothingTargetSeconds.store ((double) timeSeconds, std::memory_order_release);
        smoothingDirty.store (true, std::memory_order_release);
    }

    //==========================================================================
    // Procesamiento (AudioBuffer)
    void process (juce::AudioBuffer<SampleType>& buffer) noexcept
    {
        juce::dsp::AudioBlock<SampleType> block (buffer);
        process (block);
    }

    // Procesamiento (AudioBlock) - recomendado para chain/grafo
    void process (juce::dsp::AudioBlock<SampleType>& block) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
            return;

        const int numSamples = (int) block.getNumSamples();
        const int numCh      = (int) block.getNumChannels();

        if (numSamples <= 0 || numCh <= 0)
            return;

        // 1) Aplicar cambios de targets (audio thread, thread-safe)
        applyTargetsIfNeeded();

        // 2) Optimización de silencio (≈ -160 dB)
        const auto cur = smoother.getCurrentValue();
        const auto tgt = smoother.getTargetValue();

        if (cur < minimumGain && tgt < minimumGain)
        {
            block.clear();
            return;
        }

        // 3) Ruta rápida: sin smoothing
        if (! smoother.isSmoothing())
        {
            const auto g = tgt;

            // Unity bypass (0 dB)
            if (juce::approximatelyEqual (g, SampleType (1)))
                return;

            for (int ch = 0; ch < numCh; ++ch)
            {
                auto* data = block.getChannelPointer ((size_t) ch);
                juce::FloatVectorOperations::multiply (data, g, numSamples);
            }

            return;
        }

        // 4) Ruta precisa: smoothing activo (misma ganancia por muestra en todos los canales)
        for (int i = 0; i < numSamples; ++i)
        {
            const auto g = smoother.getNextValue();

            for (int ch = 0; ch < numCh; ++ch)
                block.getChannelPointer ((size_t) ch)[i] *= g;
        }
    }

protected:
    //==========================================================================
    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        if (spec.sampleRate <= 0.0)
            return;

        // Snapshot inicial desde atomics (por si setGain/setSmoothing se llamó antes)
        const auto g = (SampleType) gainTargetLinear.load (std::memory_order_acquire);
        const auto t = smoothingTargetSeconds.load (std::memory_order_acquire);

        smoother.reset (spec.sampleRate, (double) juce::jmax ((SampleType) t, SampleType (0)));
        smoother.setCurrentAndTargetValue (juce::jmax (g, SampleType (0)));

        // Limpia flags
        gainDirty.store (false, std::memory_order_release);
        smoothingDirty.store (false, std::memory_order_release);

        reset();
    }

private:
    //==========================================================================
    void applyTargetsIfNeeded() noexcept
    {
        // Smoothing time change: requiere reset() del smoother (hazlo en audio thread)
        if (smoothingDirty.exchange (false, std::memory_order_acq_rel))
        {
            const auto sr = getSampleRate();
            const auto newTime = smoothingTargetSeconds.load (std::memory_order_acquire);

            const auto currentVal = smoother.getCurrentValue();
            const auto targetVal  = smoother.getTargetValue();

            smoother.reset (sr, (double) juce::jmax ((SampleType) newTime, SampleType (0)));

            // Restaurar trayectoria sin saltos
            smoother.setCurrentAndTargetValue (currentVal);
            smoother.setTargetValue (targetVal);
        }

        // Gain target change
        if (gainDirty.exchange (false, std::memory_order_acq_rel))
        {
            const auto g = (SampleType) gainTargetLinear.load (std::memory_order_acquire);
            smoother.setTargetValue (juce::jmax (g, SampleType (0)));
        }
    }

    //==========================================================================
    juce::SmoothedValue<SampleType, juce::ValueSmoothingTypes::Multiplicative> smoother;

    // GUI/host -> audio (thread-safe)
    std::atomic<double> gainTargetLinear { 1.0 };
    std::atomic<double> smoothingTargetSeconds { 0.02 };

    std::atomic<bool> gainDirty { false };
    std::atomic<bool> smoothingDirty { false };

    // Piso de silencio (~ -160 dB)
    static constexpr SampleType minimumGain { SampleType (1e-8) };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Gain)
};

} // namespace xb
