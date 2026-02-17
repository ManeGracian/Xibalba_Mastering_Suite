#pragma once

#include <JuceHeader.h>
#include <atomic>

#include "XC_DSPBlockBase.h"

namespace xb
{

/*
==============================================================================
    XC_DelayAligner<SampleType>

    Delay multi-canal RT-safe para alineación de latencia en ramas (Parallel/MB).
    - No allocs/locks en process()
    - setDelaySamples() thread-safe (atomic) — aplicado en audio thread por bloque
    - Usa juce::dsp::DelayLine (interp None) para precisión de samples

==============================================================================
*/

template <typename SampleType>
class XC_DelayAligner final : public DSPBlockBase
{
public:
    XC_DelayAligner() = default;
    ~XC_DelayAligner() override = default;

    // Máximo delay soportado (en samples) — define en prepare().
    void setMaxDelaySamples (int maxDelay) noexcept
    {
        maxDelaySamples = juce::jmax (0, maxDelay);
    }

    // Thread-safe: puede llamarse desde UI/host thread.
    void setDelaySamples (int delaySamplesIn) noexcept
    {
        requestedDelay.store (juce::jmax (0, delaySamplesIn), std::memory_order_release);
        dirty.store (true, std::memory_order_release);
    }

    [[nodiscard]] int getDelaySamples() const noexcept
    {
        return requestedDelay.load (std::memory_order_acquire);
    }

    void reset() noexcept override
    {
        delayLine.reset();
    }

    void process (juce::dsp::AudioBlock<SampleType>& block) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
            return;

        applyDelayIfNeeded();

        juce::dsp::ProcessContextReplacing<SampleType> ctx (block);
        delayLine.process (ctx);
    }

    [[nodiscard]] int getLatencySamples() const noexcept override
    {
        // El delay aplicado es, por definición, latencia adicional.
        return getDelaySamples();
    }

protected:
    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        if (spec.sampleRate <= 0.0 || spec.maximumBlockSize == 0 || spec.numChannels == 0)
            return;

        if (maxDelaySamples <= 0)
            maxDelaySamples = 8192; // fallback seguro

        delayLine = juce::dsp::DelayLine<SampleType, juce::dsp::DelayLineInterpolationTypes::None> (
            (int) maxDelaySamples
        );

        delayLine.prepare (spec);
        delayLine.setDelay (0.0f);
        reset();

        dirty.store (false, std::memory_order_release);
    }

private:
    void applyDelayIfNeeded() noexcept
    {
        if (!dirty.exchange (false, std::memory_order_acq_rel))
            return;

        const int d = requestedDelay.load (std::memory_order_acquire);
        const int clamped = juce::jlimit (0, maxDelaySamples, d);

        delayLine.setDelay ((SampleType) clamped);
    }

    int maxDelaySamples { 8192 };

    std::atomic<int>  requestedDelay { 0 };
    std::atomic<bool> dirty { true };

    juce::dsp::DelayLine<SampleType, juce::dsp::DelayLineInterpolationTypes::None> delayLine;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_DelayAligner)
};

} // namespace xb
