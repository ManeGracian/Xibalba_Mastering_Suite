/*
  ==============================================================================

    XC_Filters.h
    Updated: 2026-02-16 20:45 (local)
    Author:  Mane / Xibalba Studios

    Purpose:
    - State Variable Filter (SVF) TPT (Topology Preserving Transform).
    - Basado en el enfoque de Andrew Simper.

    FIX IMPORTANT:
    - Corrección crítica en el cálculo de v1/v2:
        v1 = a1*z1 + a2*v3
        v2 = z2 + a2*z1 + a3*v3
      (antes estaban cruzados, provocando respuesta incorrecta y problemas al sumar LR4)

    RT-Safety:
    - No allocs / no locks en process()
    - Setters thread-safe via atomics; audio thread aplica targets a smoothers.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <vector>
#include <cmath>

#include "XC_DSPBlockBase.h"

namespace xb
{

template <typename SampleType>
class Filters final : public DSPBlockBase
{
public:
    enum class FilterType : int
    {
        LowPass = 0,
        HighPass,
        BandPass,
        Notch,
        Peak,
        LowShelf,
        HighShelf
    };

    Filters() = default;
    ~Filters() override = default;

    //==========================================================================
    void reset() noexcept override
    {
        std::fill (s1.begin(), s1.end(), 0.0);
        std::fill (s2.begin(), s2.end(), 0.0);

        cutoffSmoother.setCurrentAndTargetValue (cutoffSmoother.getTargetValue());
        resonanceSmoother.setCurrentAndTargetValue (resonanceSmoother.getTargetValue());
        gainSmoother.setCurrentAndTargetValue (gainSmoother.getTargetValue());
    }

    //==========================================================================
    // SETTERS (Thread-safe)
    void setType (FilterType newType) noexcept
    {
        typeTarget.store (newType, std::memory_order_release);
    }

    void setCutoffHz (SampleType newCutoffHz) noexcept
    {
        cutoffTargetHz.store ((double) newCutoffHz, std::memory_order_release);
    }

    void setResonanceQ (SampleType newQ) noexcept
    {
        resonanceTargetQ.store ((double) newQ, std::memory_order_release);
    }

    void setGainDb (SampleType newGainDb) noexcept
    {
        gainTargetDb.store ((double) newGainDb, std::memory_order_release);
    }

    //==========================================================================
    void process (juce::AudioBuffer<SampleType>& buffer) noexcept
    {
        juce::dsp::AudioBlock<SampleType> block (buffer);
        process (block);
    }

    void process (juce::dsp::AudioBlock<SampleType>& block) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
            return;

        const int numSamples = (int) block.getNumSamples();
        const int numCh      = (int) block.getNumChannels();

        if (numSamples <= 0 || numCh <= 0)
            return;

        updateSmootherTargets();

        const int stateCh = (int) s1.size();
        const int chToProcess = juce::jmin (numCh, stateCh);

        const bool smoothing =
            cutoffSmoother.isSmoothing() ||
            resonanceSmoother.isSmoothing() ||
            gainSmoother.isSmoothing();

        if (! smoothing)
            processStatic (block, numSamples, chToProcess);
        else
            processSmoothing (block, numSamples, chToProcess);
    }

protected:
    //==========================================================================
    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        s1.assign ((size_t) spec.numChannels, 0.0);
        s2.assign ((size_t) spec.numChannels, 0.0);

        cutoffSmoother.reset (spec.sampleRate, 0.015);
        resonanceSmoother.reset (spec.sampleRate, 0.015);
        gainSmoother.reset (spec.sampleRate, 0.015);

        updateSmootherTargets();
        reset();
    }

private:
    //==========================================================================
    void updateSmootherTargets() noexcept
    {
        const double sr = getSampleRate();
        if (sr <= 0.0)
            return;

        const double maxCut = sr * 0.49;
        const double minCut = 10.0;

        const double cutoffHzRaw = cutoffTargetHz.load (std::memory_order_acquire);
        const double cutoffHz    = juce::jlimit (minCut, maxCut, cutoffHzRaw);

        const double qRaw = resonanceTargetQ.load (std::memory_order_acquire);
        const double q    = juce::jmax (0.1, qRaw);

        const double gainDb = gainTargetDb.load (std::memory_order_acquire);

        cutoffSmoother.setTargetValue (cutoffHz);
        resonanceSmoother.setTargetValue (q);
        gainSmoother.setTargetValue (gainDb);

        currentType = typeTarget.load (std::memory_order_acquire);
    }

    //==========================================================================
    void processStatic (juce::dsp::AudioBlock<SampleType>& block, int numSamples, int numCh) noexcept
    {
        const double g = std::tan (juce::MathConstants<double>::pi * cutoffSmoother.getTargetValue() / getSampleRate());
        const double k = 1.0 / resonanceSmoother.getTargetValue();
        const double A = std::pow (10.0, gainSmoother.getTargetValue() / 40.0);

        double a1, a2, a3, m0, m1, m2;
        calculateCoefficients (g, k, A, a1, a2, a3, m0, m1, m2);

        for (int ch = 0; ch < numCh; ++ch)
        {
            double z1 = s1[(size_t) ch];
            double z2 = s2[(size_t) ch];

            auto* data = block.getChannelPointer ((size_t) ch);

            for (int i = 0; i < numSamples; ++i)
            {
                const double x = (double) data[i];

                // SVF TPT (Simper) - CORRECT
                const double v3 = x - z2;
                const double v1 = a1 * z1 + a2 * v3;
                const double v2 = z2 + a2 * z1 + a3 * v3;

                z1 = 2.0 * v1 - z1;
                z2 = 2.0 * v2 - z2;

                data[i] = (SampleType) (m0 * x + m1 * v1 + m2 * v2);
            }

            s1[(size_t) ch] = z1;
            s2[(size_t) ch] = z2;
        }
    }

    //==========================================================================
    void processSmoothing (juce::dsp::AudioBlock<SampleType>& block, int numSamples, int numCh) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            const double cutoffHz = cutoffSmoother.getNextValue();
            const double q        = resonanceSmoother.getNextValue();
            const double gainDb   = gainSmoother.getNextValue();

            const double g = std::tan (juce::MathConstants<double>::pi * cutoffHz / getSampleRate());
            const double k = 1.0 / q;
            const double A = std::pow (10.0, gainDb / 40.0);

            double a1, a2, a3, m0, m1, m2;
            calculateCoefficients (g, k, A, a1, a2, a3, m0, m1, m2);

            for (int ch = 0; ch < numCh; ++ch)
            {
                auto* data = block.getChannelPointer ((size_t) ch);

                const double x  = (double) data[i];
                const double z1 = s1[(size_t) ch];
                const double z2 = s2[(size_t) ch];

                // SVF TPT (Simper) - CORRECT
                const double v3 = x - z2;
                const double v1 = a1 * z1 + a2 * v3;
                const double v2 = z2 + a2 * z1 + a3 * v3;

                s1[(size_t) ch] = 2.0 * v1 - z1;
                s2[(size_t) ch] = 2.0 * v2 - z2;

                data[i] = (SampleType) (m0 * x + m1 * v1 + m2 * v2);
            }
        }
    }

    //==========================================================================
    void calculateCoefficients (double g, double k, double A,
                               double& a1, double& a2, double& a3,
                               double& m0, double& m1, double& m2) const noexcept
    {
        a1 = 1.0 / (1.0 + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;

        switch (currentType)
        {
            case FilterType::LowPass:   m0 = 0.0;   m1 = 0.0;                  m2 = 1.0;  break; // lp = v2
            case FilterType::HighPass:  m0 = 1.0;   m1 = -k;                   m2 = -1.0; break; // hp = x - k*v1 - v2
            case FilterType::BandPass:  m0 = 0.0;   m1 = 1.0;                  m2 = 0.0;  break; // bp = v1
            case FilterType::Notch:     m0 = 1.0;   m1 = -k;                   m2 = 0.0;  break;
            case FilterType::Peak:      m0 = 1.0;   m1 = k * (A * A - 1.0);    m2 = 0.0;  break;
            case FilterType::LowShelf:  m0 = 1.0;   m1 = k * (A - 1.0);        m2 = g * g * (A * A - 1.0); break;
            case FilterType::HighShelf: m0 = A * A; m1 = k * (1.0 - A) * A;    m2 = (1.0 - A * A); break;
            default:                    m0 = 1.0;   m1 = 0.0;                  m2 = 0.0;  break;
        }
    }

    //==========================================================================
    std::atomic<FilterType> typeTarget { FilterType::LowPass };
    std::atomic<double> cutoffTargetHz { 1000.0 };
    std::atomic<double> resonanceTargetQ { 0.7071 };
    std::atomic<double> gainTargetDb { 0.0 };

    FilterType currentType { FilterType::LowPass };

    juce::LinearSmoothedValue<double> cutoffSmoother { 1000.0 };
    juce::LinearSmoothedValue<double> resonanceSmoother { 0.7071 };
    juce::LinearSmoothedValue<double> gainSmoother { 0.0 };

    std::vector<double> s1, s2;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Filters)
};

} // namespace xb
