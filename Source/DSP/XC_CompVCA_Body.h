#pragma once

#include <JuceHeader.h>
#include <cmath>
#include <atomic>
#include "XC_DynamicsCore.h"

namespace xb
{

template <typename SampleType>
class XC_CompVCA_Body final : public XC_DynamicsCore<SampleType>
{
public:
    XC_CompVCA_Body() = default;
    ~XC_CompVCA_Body() override = default;

    void setAmount (SampleType newAmount) noexcept
    {
        const SampleType clamped =
            juce::jlimit (SampleType (0), SampleType (1), newAmount);

        amount.store ((double) clamped, std::memory_order_release);
    }

protected:

    SampleType calculateGainLinear (SampleType detectorLin) const noexcept override
    {
        const SampleType aRaw =
            (SampleType) amount.load (std::memory_order_acquire);

        if (aRaw <= SampleType (0.0001))
            return SampleType (1);

        // Curva no lineal
        const SampleType a = std::pow (aRaw, SampleType (1.8));

        // Mapping interno
        const SampleType thresholdDb =
            juce::jmap (a, SampleType (0), SampleType (1),
                        SampleType (-8.0), SampleType (-32.0));

        const SampleType ratio =
            juce::jmap (a, SampleType (0), SampleType (1),
                        SampleType (1.2), SampleType (3.2));

        const SampleType slope =
            SampleType (1) - SampleType (1) / ratio;

        const SampleType kneeWidthDb =
            juce::jmap (a, SampleType (0), SampleType (1),
                        SampleType (8.0), SampleType (12.0));

        detectorLin = juce::jmax (detectorLin, SampleType (1e-12));
        const SampleType inputDb =
            juce::Decibels::gainToDecibels (detectorLin);

        const SampleType overDb = inputDb - thresholdDb;
        const SampleType halfKnee = kneeWidthDb * SampleType (0.5);

        SampleType gainReductionDb = SampleType (0);

        if (overDb < -halfKnee)
        {
            gainReductionDb = SampleType (0);
        }
        else if (std::abs (overDb) <= halfKnee)
        {
            const SampleType kneeStart = overDb + halfKnee;
            const SampleType overKnee =
                (kneeStart * kneeStart) /
                (kneeWidthDb * SampleType (2));

            gainReductionDb = -overKnee * slope;
        }
        else
        {
            gainReductionDb = -overDb * slope;
        }

        if (gainReductionDb >= SampleType (0))
            return SampleType (1);

        return juce::Decibels::decibelsToGain (gainReductionDb);
    }

private:
    std::atomic<double> amount { 0.0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_CompVCA_Body)
};

} // namespace xb
