#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cmath>

#include "XC_DSPBlockBase.h"
#include "XC_Oversampler.h"

namespace xb
{

/*
==============================================================================
    XC_Sat

    Saturación base abstracta:
    - Oversampling via xb::Oversampler<float> (framework-level, RT-safe)
    - DC blocker por canal post-nonlinearity
    - Parámetros thread-safe (Drive/Trim/Mix): setters -> atomics, audio -> snapshot

==============================================================================
*/

class XC_Sat : public DSPBlockBase
{
public:
    using OversamplingMode = xb::Oversampler<float>::Mode;

    XC_Sat() = default;
    ~XC_Sat() override = default;

    //==========================================================================
    void reset() noexcept override
    {
        oversampler.reset();

        if (dcState != nullptr) juce::FloatVectorOperations::clear (dcState.get(), numChannelsPrepared);
        if (dcLast  != nullptr) juce::FloatVectorOperations::clear (dcLast.get(),  numChannelsPrepared);
    }

    //==========================================================================
    // Thread-safe setters (UI/host thread)
    void setDrive (float driveDb) noexcept
    {
        driveTargetLinear.store (juce::Decibels::decibelsToGain (driveDb), std::memory_order_release);
        paramsDirty.store (true, std::memory_order_release);
    }

    void setOutputTrim (float trimDb) noexcept
    {
        trimTargetLinear.store (juce::Decibels::decibelsToGain (trimDb), std::memory_order_release);
        paramsDirty.store (true, std::memory_order_release);
    }

    void setMix (float mix01) noexcept
    {
        mixTarget.store (juce::jlimit (0.0f, 1.0f, mix01), std::memory_order_release);
        paramsDirty.store (true, std::memory_order_release);
    }

    void setOversamplingMode (OversamplingMode newMode) noexcept
    {
        oversampler.setMode (newMode);
    }

    [[nodiscard]] float getDriveLinear() const noexcept
    {
        return driveTargetLinear.load (std::memory_order_acquire);
    }

    //==========================================================================
    void process (juce::AudioBuffer<float>& buffer) noexcept
    {
        juce::dsp::AudioBlock<float> block (buffer);
        process (block);
    }

    void process (juce::dsp::AudioBlock<float>& block) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
            return;

        const int chs = (int) block.getNumChannels();
        const int n   = (int) block.getNumSamples();
        if (chs <= 0 || n <= 0)
            return;

        applyTargetsIfNeeded();

        // Fast paths
        if (mix <= 0.000001f)
        {
            applyTrim (block);
            return;
        }

        if (driveLinear <= 1.0001f && mix >= 0.999f)
        {
            applyTrim (block);
            return;
        }

        // Oversampling (Off/bypass transparente)
        oversampler.processOversampled (block, [&] (auto& osBlock) noexcept
        {
            processBlockNoOS (osBlock);
        });

        applyTrim (block);
    }

    //==========================================================================
    [[nodiscard]] int getLatencySamples() const noexcept override
    {
        return oversampler.getLatencySamples();
    }

protected:
    virtual float applySaturationMath (float x) noexcept = 0;

    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        numChannelsPrepared = (int) spec.numChannels;

        oversampler.prepare (spec);

        dcState.allocate ((size_t) numChannelsPrepared, true);
        dcLast .allocate ((size_t) numChannelsPrepared, true);
        updateDCCoefficient (spec.sampleRate);

        applyTargetsIfNeeded (true);
        reset();
    }

private:
    //==========================================================================
    void applyTargetsIfNeeded (bool force = false) noexcept
    {
        if (!force && !paramsDirty.exchange (false, std::memory_order_acq_rel))
            return;

        driveLinear = juce::jmax (driveTargetLinear.load (std::memory_order_acquire), 0.0f);
        trimLinear  = juce::jmax (trimTargetLinear.load  (std::memory_order_acquire), 0.0f);
        mix         = juce::jlimit (0.0f, 1.0f, mixTarget.load (std::memory_order_acquire));
    }

    //==========================================================================
    void processBlockNoOS (juce::dsp::AudioBlock<float>& block) noexcept
    {
        const int chs = (int) block.getNumChannels();
        const int n   = (int) block.getNumSamples();

        const bool hasDrive = (driveLinear > 1.0001f);

        for (int ch = 0; ch < chs; ++ch)
        {
            auto* data = block.getChannelPointer ((size_t) ch);

            const int ctx = (numChannelsPrepared > 0) ? (ch % numChannelsPrepared) : 0;
            float z1 = dcState[ctx];
            float x1 = dcLast [ctx];

            for (int i = 0; i < n; ++i)
            {
                const float dry = data[i];
                float wet = dry;

                if (hasDrive)
                    wet = applySaturationMath (dry * driveLinear);

                const float y = wet - x1 + dcCoeff * z1;
                x1 = wet;
                z1 = y;

                data[i] = dry * (1.0f - mix) + y * mix;
            }

            dcState[ctx] = z1;
            dcLast [ctx] = x1;
        }
    }

    //==========================================================================
    void applyTrim (juce::dsp::AudioBlock<float>& block) noexcept
    {
        if (juce::approximatelyEqual (trimLinear, 1.0f))
            return;

        const int chs = (int) block.getNumChannels();
        const int n   = (int) block.getNumSamples();

        for (int ch = 0; ch < chs; ++ch)
            juce::FloatVectorOperations::multiply (block.getChannelPointer ((size_t) ch), trimLinear, n);
    }

    //==========================================================================
    void updateDCCoefficient (double sr) noexcept
    {
        constexpr float fc = 10.0f;
        const auto w = 2.0f * juce::MathConstants<float>::pi * fc;
        dcCoeff = std::exp (-w / (float) sr);
    }

    //==========================================================================
    int numChannelsPrepared { 2 };

    // Snapshot (audio thread)
    float driveLinear { 1.0f };
    float trimLinear  { 1.0f };
    float mix         { 1.0f };

    // Targets (UI/host -> audio thread)
    std::atomic<float> driveTargetLinear { 1.0f };
    std::atomic<float> trimTargetLinear  { 1.0f };
    std::atomic<float> mixTarget         { 1.0f };
    std::atomic<bool>  paramsDirty       { true };

    // Oversampling wrapper
    xb::Oversampler<float> oversampler;

    // DC blocker
    juce::HeapBlock<float> dcState;
    juce::HeapBlock<float> dcLast;
    float dcCoeff { 0.995f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_Sat)
};

} // namespace xb
