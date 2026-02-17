#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cmath>
#include "../XC_DSPBlockBase.h"
#include "../XC_Oversampler.h"

namespace xb
{

class XC_RS124_ModelCore final : public DSPBlockBase
{
public:
    enum class DetectorMode
    {
        LinkedMono = 0,
        DualMono   = 1
    };

    XC_RS124_ModelCore() = default;
    ~XC_RS124_ModelCore() override = default;

    //==============================================================
    void setAmount (float amt) noexcept
    {
        amt = juce::jlimit (0.0f, 1.0f, amt);
        amount.store (amt, std::memory_order_release);
    }

    void setOversamplingMode (xb::Oversampler<float>::Mode mode) noexcept
    {
        oversampler.setMode (mode);
    }

    void setDetectorMode (DetectorMode mode) noexcept
    {
        detectorMode.store (mode, std::memory_order_release);
    }

    [[nodiscard]] float getCurrentGainReductionDb() const noexcept
    {
        return currentGRdB.load (std::memory_order_acquire);
    }

    //==============================================================
    void reset() noexcept override
    {
        oversampler.reset();
        envL = envR = 0.0f;
        currentGRdB.store (0.0f, std::memory_order_release);
    }

    //==============================================================
    void process (juce::AudioBuffer<float>& buffer) noexcept
    {
        const float amt = amount.load (std::memory_order_acquire);

        if (amt <= 0.000001f)
            return; // Bypass real, 0 latencia, 0 color

        juce::dsp::AudioBlock<float> block (buffer);

        oversampler.processOversampled (block,
        [&](juce::dsp::AudioBlock<float>& osBlock)
        {
            processBlock (osBlock, amt);
        });
    }

    [[nodiscard]] int getLatencySamples() const noexcept override
    {
        if (amount.load(std::memory_order_acquire) <= 0.000001f)
            return 0;

        return oversampler.getLatencySamples();
    }

protected:

    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        oversampler.prepare (spec);
        sampleRate = spec.sampleRate;
        reset();
    }

private:

    //==============================================================
    void processBlock (juce::dsp::AudioBlock<float>& block, float amt) noexcept
    {
        const int chs = (int) block.getNumChannels();
        const int n   = (int) block.getNumSamples();

        const float attackMs  = 30.0f;
        const float releaseMs = 250.0f;

        const float attackCoeff  = calcCoeff (attackMs);
        const float baseRelease  = calcCoeff (releaseMs);

        const auto mode = detectorMode.load (std::memory_order_acquire);

        for (int i = 0; i < n; ++i)
        {
            float monoEnergy = 0.0f;

            if (mode == DetectorMode::LinkedMono)
            {
                for (int ch = 0; ch < chs; ++ch)
                {
                    const float x = block.getChannelPointer (ch)[i];
                    monoEnergy += x * x;
                }

                monoEnergy /= (float) chs;
                updateEnvelope (monoEnergy, envL, attackCoeff, baseRelease);
            }

            for (int ch = 0; ch < chs; ++ch)
            {
                float x = block.getChannelPointer (ch)[i];

                float env;

                if (mode == DetectorMode::LinkedMono)
                {
                    env = std::sqrt (juce::jmax (envL, 0.0f));
                }
                else
                {
                    float energy = x * x;

                    if (ch == 0)
                        updateEnvelope (energy, envL, attackCoeff, baseRelease);
                    else
                        updateEnvelope (energy, envR, attackCoeff, baseRelease);

                    env = std::sqrt (juce::jmax ((ch == 0 ? envL : envR), 0.0f));
                }

                // Vari-mu perceptual curve
                const float drive = amt * 4.0f;
                const float control = env * drive;

                float gain = 1.0f / (1.0f + control + 0.5f * control * control);

                // Program dependent release feel
                const float slowFactor = 1.0f + env * 2.0f;
                gain = std::pow (gain, slowFactor);

                float y = x * gain;

                block.getChannelPointer (ch)[i] = y;

                const float gr = juce::jlimit (0.0001f, 1.0f, gain);
                currentGRdB.store (
                    juce::Decibels::gainToDecibels (gr),
                    std::memory_order_release);
            }
        }
    }

    //==============================================================
    void updateEnvelope (float input,
                         float& env,
                         float attackCoeff,
                         float releaseCoeff) noexcept
    {
        if (input > env)
            env += attackCoeff * (input - env);
        else
            env += releaseCoeff * (input - env);
    }

    float calcCoeff (float ms) const noexcept
    {
        const float T = ms * 0.001f;
        return 1.0f - std::exp (-1.0f / (T * (float) sampleRate));
    }

    //==============================================================
    xb::Oversampler<float> oversampler;

    std::atomic<float> amount { 0.6f };
    std::atomic<DetectorMode> detectorMode { DetectorMode::LinkedMono };
    std::atomic<float> currentGRdB { 0.0f };

    double sampleRate { 44100.0 };

    float envL { 0.0f };
    float envR { 0.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_RS124_ModelCore)
};

} // namespace xb

