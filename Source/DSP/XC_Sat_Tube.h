#pragma once

#include <JuceHeader.h>
#include <cmath>
#include <atomic>

#include "XC_Sat.h"

namespace xb
{

/*
    Tube-style saturation (triode-ish) using asymmetric transfer and smooth polarity blending.

    - Bias: controls asymmetry. Range: [0.0 .. 0.5]
    - Thread-safe: setBias() writes atomics; audio reads atomics.
*/

class XC_Sat_Tube final : public XC_Sat
{
public:
    XC_Sat_Tube() = default;
    ~XC_Sat_Tube() override = default;

    void setBias (float newBias) noexcept
    {
        newBias = juce::jlimit (0.0f, 0.5f, newBias);

        const float newOffset = calculateBiasOffset (newBias);

        bias.store (newBias, std::memory_order_relaxed);
        biasOffset.store (newOffset, std::memory_order_relaxed);
    }

    [[nodiscard]] float getBias() const noexcept
    {
        return bias.load (std::memory_order_relaxed);
    }

protected:
    float applySaturationMath (float x) noexcept override final
    {
        const float b  = bias.load (std::memory_order_relaxed);
        const float bo = biasOffset.load (std::memory_order_relaxed);

        constexpr float headroom = 0.5f;
        const float input = x * headroom;

        const float xBiased = input + b;
        const float limited = juce::jlimit (-10.0f, 10.0f, xBiased);

        const float expCurve  = 1.0f - std::exp (-limited);
        const float tanhCurve = std::tanh (limited);

        const float blend = 0.5f + 0.5f * std::tanh (limited * 3.5f);
        float y = (expCurve * blend) + (tanhCurve * (1.0f - blend));

        y -= bo;
        return y;
    }

private:
    static float calculateBiasOffset (float b) noexcept
    {
        const float limited = juce::jlimit (-10.0f, 10.0f, b);

        const float e = 1.0f - std::exp (-limited);
        const float t = std::tanh (limited);
        const float p = 0.5f + 0.5f * std::tanh (limited * 3.5f);

        return (e * p) + (t * (1.0f - p));
    }

    std::atomic<float> bias { 0.15f };
    std::atomic<float> biasOffset { calculateBiasOffset (0.15f) }; // ✅ recomendado

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_Sat_Tube)
};

} // namespace xb
