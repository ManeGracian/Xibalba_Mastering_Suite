/*
  ==============================================================================

    XC_CompVCA.h
    Created: 15 Feb 2026
    Author:  Mane / Xibalba Studios

    Description:
    VCA (Voltage Controlled Amplifier) style compression.
    Clean, transparent, and predictable gain-reduction curve.

    Features:
    - Variable soft knee (quadratic transition)
    - Deterministic gain computer (no state)
    - Inherits detector/lookahead/ballistics from XC_DynamicsCore

    RT-Safety:
    - No allocations, no locks.
    - calculateGainLinear() is pure math, noexcept.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <cmath>
#include "XC_DynamicsCore.h"

namespace xb
{

template <typename SampleType>
class XC_CompVCA final : public XC_DynamicsCore<SampleType>
{
public:
    XC_CompVCA() = default;
    ~XC_CompVCA() override = default;

protected:
    //==============================================================================
    // Gain computer (VCA character)
    //
    // Input : detectorLin (envelope) in linear gain domain, > 0
    // Output: linear gain factor [0..1], typically <= 1 (no upward compression)
    //
    // Notes:
    // - Uses this->thresholdDb, this->ratio, this->slope, this->kneeWidth
    // - this->slope is precomputed in XC_DynamicsCore as: slope = 1 - 1/ratio
    //   which is positive for ratio > 1.
    //==============================================================================
    SampleType calculateGainLinear (SampleType detectorLin) const noexcept override
    {
        // Ratio <= 1 => no compression
        if (this->ratio <= SampleType (1.001))
            return SampleType (1);

        // Convert detector to dB safely:
        // clamp to avoid -inf at/near zero. This is cheap and stable.
        detectorLin = juce::jmax (detectorLin, SampleType (1e-12));
        const SampleType inputDb = juce::Decibels::gainToDecibels (detectorLin);

        const SampleType thresholdDb = this->thresholdDb;
        const SampleType kneeWidthDb = juce::jmax (this->kneeWidth, SampleType (0));

        const SampleType overDb   = inputDb - thresholdDb;
        const SampleType slope    = this->slope; // = 1 - 1/ratio (positive)

        // If knee == 0: hard-knee classic
        if (kneeWidthDb <= SampleType (0))
        {
            if (overDb <= SampleType (0))
                return SampleType (1);

            // gainReductionDb is negative (attenuation)
            const SampleType gainReductionDb = -overDb * slope;
            return juce::Decibels::decibelsToGain (gainReductionDb);
        }

        const SampleType halfKnee = kneeWidthDb * SampleType (0.5);

        // Zone A: below knee start -> no compression
        if (overDb < -halfKnee)
            return SampleType (1);

        SampleType gainReductionDb = SampleType (0);

        // Zone B: inside knee -> quadratic transition
        if (std::abs (overDb) <= halfKnee)
        {
            // Quadratic soft-knee:
            // y = (x + w/2)^2 / (2w)
            // where x = overDb, w = kneeWidthDb
            const SampleType kneeStart = overDb + halfKnee;
            const SampleType overKnee  = (kneeStart * kneeStart) / (kneeWidthDb * SampleType (2));

            gainReductionDb = -overKnee * slope;
        }
        // Zone C: above knee end -> full compression line
        else
        {
            gainReductionDb = -overDb * slope;
        }

        // Safety: never boost (should already be <= 0)
        if (gainReductionDb >= SampleType (0))
            return SampleType (1);

        return juce::Decibels::decibelsToGain (gainReductionDb);
    }

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_CompVCA)
};

} // namespace xb
