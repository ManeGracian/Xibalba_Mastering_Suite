#pragma once

#include <JuceHeader.h>
#include <atomic>
#include "XC_DSPBlockBase.h"
#include "XC_Filters.h"

namespace xb
{

/*
==============================================================================
    xb::Crossover<SampleType>

    Linkwitz-Riley 4th Order (LR4) Crossover (24 dB/oct)
    - Rama Low:  2x Butterworth 2nd order LP (Q = 1/sqrt(2))
    - Rama High: 2x Butterworth 2nd order HP (Q = 1/sqrt(2))
    - Suma coherente en fase (si recombinas Low + High)

    RT-Safety:
    - No allocs / no locks en process()
    - setCutoffHz() es thread-safe (UI/host thread) -> atomic
    - El audio thread aplica cambios al inicio de cada bloque

    Channel-agnostic:
    - Funciona con 1..N canales, dimensionado en prepare()
==============================================================================
*/

template <typename SampleType>
class Crossover final : public DSPBlockBase
{
public:
    Crossover() = default;
    ~Crossover() override = default;

    //==========================================================================
    void reset() noexcept override
    {
        lowStage1.reset();
        lowStage2.reset();
        highStage1.reset();
        highStage2.reset();

        lastAppliedCutoffHz = -1.0;
    }

    //==========================================================================
    // Thread-safe setter (puede llamarse desde UI/host thread)
    void setCutoffHz (SampleType newFreqHz) noexcept
    {
        cutoffTargetHz.store ((double) newFreqHz, std::memory_order_release);
    }

    [[nodiscard]] SampleType getCutoffHz() const noexcept
    {
        return (SampleType) cutoffTargetHz.load (std::memory_order_acquire);
    }

    //==========================================================================
    // Procesamiento: AudioBuffer (split a dos buffers ya prealocados)
    void process (const juce::AudioBuffer<SampleType>& input,
                  juce::AudioBuffer<SampleType>& lowBuffer,
                  juce::AudioBuffer<SampleType>& highBuffer) noexcept
    {
        juce::dsp::AudioBlock<const SampleType> inBlock (input);
        juce::dsp::AudioBlock<SampleType> lowBlock (lowBuffer);
        juce::dsp::AudioBlock<SampleType> highBlock (highBuffer);

        process (inBlock, lowBlock, highBlock);
    }

    //==========================================================================
    // Procesamiento: AudioBlock (recomendado para motor de ruteo/grafo)
    void process (const juce::dsp::AudioBlock<const SampleType>& input,
                  juce::dsp::AudioBlock<SampleType>& lowOut,
                  juce::dsp::AudioBlock<SampleType>& highOut) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
        {
            // Bypass coherente: lowOut=input, highOut=0 (o copia también, tú decides).
            copyInputToOutputs (input, lowOut, highOut);
            highOut.clear();
            return;
        }

        const int numSamples = (int) input.getNumSamples();
        const int inCh       = (int) input.getNumChannels();
        const int lowCh      = (int) lowOut.getNumChannels();
        const int highCh     = (int) highOut.getNumChannels();

        if (numSamples <= 0 || inCh <= 0)
            return;

        // Defensa: esperamos que low/high tengan >= canales de input.
        const int chToCopy = juce::jmin (inCh, juce::jmin (lowCh, highCh));
        if (chToCopy <= 0)
            return;

        // 1) Aplicar cutoff pendiente (audio-thread)
        applyCutoffIfNeeded();

        // 2) Copiar input -> low/high (sin allocs)
        for (int ch = 0; ch < chToCopy; ++ch)
        {
            juce::FloatVectorOperations::copy (lowOut.getChannelPointer ((size_t) ch),
                                               input.getChannelPointer ((size_t) ch),
                                               numSamples);

            juce::FloatVectorOperations::copy (highOut.getChannelPointer ((size_t) ch),
                                               input.getChannelPointer ((size_t) ch),
                                               numSamples);
        }

        // Si low/high tienen canales extra, los limpiamos para evitar basura
        for (int ch = chToCopy; ch < lowCh; ++ch)
            juce::FloatVectorOperations::clear (lowOut.getChannelPointer ((size_t) ch), numSamples);

        for (int ch = chToCopy; ch < highCh; ++ch)
            juce::FloatVectorOperations::clear (highOut.getChannelPointer ((size_t) ch), numSamples);

        // 3) Filtrado LR4 (24 dB/oct) por rama
        // LOW: LP -> LP
        lowStage1.process (lowOut);
        lowStage2.process (lowOut);

        // HIGH: HP -> HP
        highStage1.process (highOut);
        highStage2.process (highOut);
    }

protected:
    //==========================================================================
    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        if (spec.sampleRate <= 0.0 || spec.maximumBlockSize == 0 || spec.numChannels == 0)
            return;

        // Configuración base de los 4 SVF
        // Butterworth 2nd order: Q = 1/sqrt(2)
        constexpr SampleType butterworthQ = (SampleType) 0.7071067811865475;

        // --- Rama LOW (LP LR4) ---
        lowStage1.prepare (spec);
        lowStage1.setType (xb::Filters<SampleType>::FilterType::LowPass);
        lowStage1.setResonanceQ (butterworthQ);

        lowStage2.prepare (spec);
        lowStage2.setType (xb::Filters<SampleType>::FilterType::LowPass);
        lowStage2.setResonanceQ (butterworthQ);

        // --- Rama HIGH (HP LR4) ---
        highStage1.prepare (spec);
        highStage1.setType (xb::Filters<SampleType>::FilterType::HighPass);
        highStage1.setResonanceQ (butterworthQ);

        highStage2.prepare (spec);
        highStage2.setType (xb::Filters<SampleType>::FilterType::HighPass);
        highStage2.setResonanceQ (butterworthQ);

        lastAppliedCutoffHz = -1.0;
        applyCutoffIfNeeded(); // aplica cutoff inicial (clamp con sr real)

        reset();
    }

private:
    //==========================================================================
    void applyCutoffIfNeeded() noexcept
    {
        const double sr = getSampleRate();
        if (sr <= 0.0)
            return;

        const double raw = cutoffTargetHz.load (std::memory_order_acquire);

        // Clamps defensivos
        const double minHz = 10.0;
        const double maxHz = sr * 0.49; // evita inestabilidad cerca de Nyquist
        const double cutoffHz = juce::jlimit (minHz, maxHz, raw);

        // Evitar churn si no cambió (en Hz)
        if (juce::approximatelyEqual (cutoffHz, lastAppliedCutoffHz))
            return;

        lastAppliedCutoffHz = cutoffHz;

        lowStage1.setCutoffHz ((SampleType) cutoffHz);
        lowStage2.setCutoffHz ((SampleType) cutoffHz);
        highStage1.setCutoffHz ((SampleType) cutoffHz);
        highStage2.setCutoffHz ((SampleType) cutoffHz);
    }

    //==========================================================================
    static void copyInputToOutputs (const juce::dsp::AudioBlock<const SampleType>& input,
                                    juce::dsp::AudioBlock<SampleType>& lowOut,
                                    juce::dsp::AudioBlock<SampleType>& highOut) noexcept
    {
        const int n   = (int) input.getNumSamples();
        const int inCh = (int) input.getNumChannels();
        const int lowCh = (int) lowOut.getNumChannels();
        const int highCh = (int) highOut.getNumChannels();

        const int chToCopy = juce::jmin (inCh, juce::jmin (lowCh, highCh));
        for (int ch = 0; ch < chToCopy; ++ch)
        {
            juce::FloatVectorOperations::copy (lowOut.getChannelPointer ((size_t) ch),
                                               input.getChannelPointer ((size_t) ch),
                                               n);

            juce::FloatVectorOperations::copy (highOut.getChannelPointer ((size_t) ch),
                                               input.getChannelPointer ((size_t) ch),
                                               n);
        }
    }

    //==========================================================================
    xb::Filters<SampleType> lowStage1, lowStage2;
    xb::Filters<SampleType> highStage1, highStage2;

    // Thread-safe target para cutoff
    std::atomic<double> cutoffTargetHz { 500.0 };

    // Snapshot (audio thread) para evitar re-setear cada bloque
    double lastAppliedCutoffHz { -1.0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Crossover)
};

} // namespace xb
