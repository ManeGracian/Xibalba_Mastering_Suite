#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <array>
#include <memory>

#include "XC_DSPBlockBase.h"

namespace xb
{

/*
==============================================================================
    xb::Oversampler<SampleType>

    Oversampling wrapper RT-safe y “lego-friendly” para JUCE dsp::Oversampling.

    - RT-safe: no allocs/initProcessing en process()
    - Thread-safe: setMode() vía atomics
    - Channel-agnostic: preparado con spec.numChannels
    - API segura: processOversampled(block, fn) encapsula up/down + bypass

==============================================================================
*/

template <typename SampleType>
class Oversampler final : public DSPBlockBase
{
public:
    enum class Mode : int
    {
        Off = 0,
        x2  = 1,
        x4  = 2,
        x8  = 3
    };

    Oversampler() = default;
    ~Oversampler() override = default;

    //==========================================================================
    void reset() noexcept override
    {
        for (auto& os : oversamplers)
            if (os != nullptr)
                os->reset();
    }

    //==========================================================================
    // Thread-safe (UI/host thread)
    void setMode (Mode newMode) noexcept
    {
        requestedMode.store (newMode, std::memory_order_release);
    }

    [[nodiscard]] Mode getMode() const noexcept
    {
        return requestedMode.load (std::memory_order_acquire);
    }

    //==========================================================================
    template <typename ProcessFn>
    void processOversampled (juce::dsp::AudioBlock<SampleType>& block, ProcessFn&& fn) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
        {
            fn (block);
            return;
        }

        auto* os = getOversamplerForCurrentMode();
        if (os == nullptr)
        {
            fn (block);
            return;
        }

        auto osBlock = os->processSamplesUp (block);
        fn (osBlock);
        os->processSamplesDown (block);
    }

    //==========================================================================
    [[nodiscard]] int getLatencySamples() const noexcept override
    {
        if (auto* os = getOversamplerForCurrentMode())
            return (int) os->getLatencyInSamples();
        return 0;
    }

    [[nodiscard]] int getMaxLatencySamples() const noexcept
    {
        int maxLat = 0;
        for (const auto& os : oversamplers)
            if (os != nullptr)
                maxLat = juce::jmax (maxLat, (int) os->getLatencyInSamples());
        return maxLat;
    }

protected:
    //==========================================================================
    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        if (spec.sampleRate <= 0.0 || spec.maximumBlockSize == 0 || spec.numChannels == 0)
            return;

        numChannelsPrepared  = (int) spec.numChannels;
        maxBlockSizePrepared = (int) spec.maximumBlockSize;

        oversamplers.fill (nullptr);

        createOversampler (Mode::x2, 1); // 2x
        createOversampler (Mode::x4, 2); // 4x
        createOversampler (Mode::x8, 3); // 8x

        reset();
    }

private:
    using JUCEOS = juce::dsp::Oversampling<SampleType>;

    static constexpr auto filterType =
        JUCEOS::filterHalfBandPolyphaseIIR;

    void createOversampler (Mode mode, int numStages)
    {
        auto os = std::make_unique<JUCEOS> (
            numChannelsPrepared,
            (size_t) numStages,
            filterType,
            true /* isLatencyCompensated */
        );

        os->initProcessing ((size_t) maxBlockSizePrepared);
        oversamplers[(size_t) mode] = std::move (os);
    }

    JUCEOS* getOversamplerForCurrentMode() const noexcept
    {
        const auto mode = requestedMode.load (std::memory_order_acquire);
        if (mode == Mode::Off)
            return nullptr;

        auto& ptr = oversamplers[(size_t) mode];
        return ptr.get();
    }

    std::atomic<Mode> requestedMode { Mode::Off };
    std::array<std::unique_ptr<JUCEOS>, 4> oversamplers;

    int numChannelsPrepared  { 0 };
    int maxBlockSizePrepared { 0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Oversampler)
};

} // namespace xb
