#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <array>
#include <memory>

#include "XC_DSPBlockBase.h"

namespace xb
{

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

    //==============================================================
    void reset() noexcept override
    {
        for (auto& os : oversamplers)
            if (os)
                os->reset();
    }

    //==============================================================
    void setMode (Mode newMode) noexcept
    {
        requestedMode.store (newMode, std::memory_order_release);
    }

    [[nodiscard]] Mode getMode() const noexcept
    {
        return requestedMode.load (std::memory_order_acquire);
    }

    //==============================================================
    template <typename ProcessFn>
    void processOversampled (juce::dsp::AudioBlock<SampleType>& block,
                             ProcessFn&& fn) noexcept
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

    //==============================================================
    [[nodiscard]] int getLatencySamples() const noexcept override
    {
        if (auto* os = getOversamplerForCurrentMode())
            return (int) os->getLatencyInSamples();

        return 0;
    }

protected:

    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        if (spec.sampleRate <= 0.0 ||
            spec.maximumBlockSize == 0 ||
            spec.numChannels == 0)
            return;

        numChannelsPrepared  = (int) spec.numChannels;
        maxBlockSizePrepared = (int) spec.maximumBlockSize;

        // 🔥 CORRECCIÓN: no usar fill() con unique_ptr
        for (auto& os : oversamplers)
            os.reset();

        createOversampler (Mode::x2, 1);
        createOversampler (Mode::x4, 2);
        createOversampler (Mode::x8, 3);

        reset();
    }

private:

    using JUCEOS = juce::dsp::Oversampling<SampleType>;

    static constexpr auto filterType =
        JUCEOS::filterHalfBandPolyphaseIIR;

    void createOversampler (Mode mode,
                            int numStages)
    {
        auto os = std::make_unique<JUCEOS>(
            numChannelsPrepared,
            (size_t) numStages,
            filterType,
            true
        );

        os->initProcessing ((size_t) maxBlockSizePrepared);

        oversamplers[(size_t) mode] = std::move (os);
    }

    JUCEOS* getOversamplerForCurrentMode() const noexcept
    {
        const auto mode =
            requestedMode.load (std::memory_order_acquire);

        if (mode == Mode::Off)
            return nullptr;

        return oversamplers[(size_t) mode].get();
    }

    std::atomic<Mode> requestedMode { Mode::Off };

    std::array<std::unique_ptr<JUCEOS>, 4> oversamplers;

    int numChannelsPrepared  { 0 };
    int maxBlockSizePrepared { 0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Oversampler)
};

} // namespace xb

