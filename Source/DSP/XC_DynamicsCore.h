#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <vector>
#include <cmath>
#include "XC_DSPBlockBase.h"

namespace xb
{

template <typename SampleType>
class XC_DynamicsCore : public DSPBlockBase
{
public:
    XC_DynamicsCore() = default;
    ~XC_DynamicsCore() override = default;

    //==========================================================================
    void reset() noexcept override
    {
        delayBuffer.clear();
        std::fill (scFilterStates.begin(), scFilterStates.end(), SampleType (0));

        envelopeState    = SampleType (0);
        currentGainState = SampleType (1);
        writePos         = 0;

        lookAheadSmoother.setCurrentAndTargetValue (SampleType (0));
    }

    //==========================================================================
    // Thread-safe setters
    void setThreshold (SampleType db) noexcept
    {
        thresholdDbTarget.store ((double) db);
    }

    void setRatio (SampleType r) noexcept
    {
        ratioTarget.store ((double) juce::jmax (SampleType (1), r));
    }

    void setKnee (SampleType db) noexcept
    {
        kneeWidthTarget.store ((double) juce::jmax (SampleType (0), db));
    }

    void setMakeupGain (SampleType db) noexcept
    {
        makeupDbTarget.store ((double) db);
    }

    void setAttack (SampleType ms) noexcept
    {
        attackMsTarget.store ((double) juce::jmax (SampleType (0.1), ms));
    }

    void setRelease (SampleType ms) noexcept
    {
        releaseMsTarget.store ((double) juce::jmax (SampleType (0.1), ms));
    }

    void setLookAhead (SampleType ms) noexcept
    {
        const auto clamped =
            juce::jlimit (SampleType (0), SampleType (maxLookAheadMs), ms);

        lookAheadMsTarget.store ((double) clamped);
    }

    void setSidechainHPF (SampleType freqHz) noexcept
    {
        scHpfHzTarget.store ((double) juce::jmax (SampleType (0), freqHz));
    }

    [[nodiscard]] SampleType getCurrentGain() const noexcept
    {
        return currentGainState;
    }

    [[nodiscard]] int getLatencySamples() const noexcept override
    {
        return (int) lookAheadSmoother.getCurrentValue();
    }

    //==========================================================================
    void process (juce::AudioBuffer<SampleType>& buffer) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
            return;

        const int numSamples = buffer.getNumSamples();
        const int inCh       = buffer.getNumChannels();

        if (numSamples <= 0 || inCh <= 0)
            return;

        const int delaySize = delayBuffer.getNumSamples();
        if (delaySize <= 1)
            return;

        updateFromTargets();

        const int preparedCh = (int) scFilterStates.size();
        const int chToProcess = juce::jmin (inCh, preparedCh);

        const SampleType* const* inputs  = buffer.getArrayOfReadPointers();
        SampleType* const*       outputs = buffer.getArrayOfWritePointers();
        SampleType* const*       delayW  = delayBuffer.getArrayOfWritePointers();
        const SampleType* const* delayR  = delayBuffer.getArrayOfReadPointers();

        constexpr SampleType gainSmoothing = SampleType (0.005);

        for (int i = 0; i < numSamples; ++i)
        {
            const SampleType currentDelay = lookAheadSmoother.getNextValue();

            SampleType maxAbs = SampleType (0);

            for (int ch = 0; ch < chToProcess; ++ch)
            {
                const SampleType x = inputs[ch][i];

                delayW[ch][writePos] = x;

                scFilterStates[(size_t) ch] +=
                    scHpfCoeff * (x - scFilterStates[(size_t) ch]);

                const SampleType detectorSignal =
                    x - scFilterStates[(size_t) ch];

                const SampleType s = std::abs (detectorSignal);
                if (s > maxAbs)
                    maxAbs = s;
            }

            if (maxAbs > envelopeState)
                envelopeState += attackCoeff  * (maxAbs - envelopeState);
            else
                envelopeState += releaseCoeff * (maxAbs - envelopeState);

            const SampleType detectorLin =
                juce::jmax (envelopeState, SampleType (1e-12));

            const SampleType rawGain =
                calculateGainLinear (detectorLin);

            currentGainState +=
                gainSmoothing * (rawGain - currentGainState);

            double readPos = (double) writePos - (double) currentDelay;
            if (readPos < 0.0)
                readPos += (double) delaySize;

            const int idxA = (int) readPos;
            const int idxB = (idxA + 1) % delaySize;
            const SampleType frac =
                (SampleType) (readPos - (double) idxA);

            for (int ch = 0; ch < chToProcess; ++ch)
            {
                const SampleType a = delayR[ch][idxA];
                const SampleType b = delayR[ch][idxB];

                const SampleType delayed =
                    a + frac * (b - a);

                outputs[ch][i] =
                    delayed * currentGainState * makeupGainLin;
            }

            writePos = (writePos + 1) % delaySize;
        }
    }

protected:
    virtual SampleType calculateGainLinear (SampleType detectorLin) const noexcept = 0;

    SampleType thresholdDb   { SampleType (0) };
    SampleType ratio         { SampleType (1) };
    SampleType slope         { SampleType (0) };
    SampleType kneeWidth     { SampleType (0) };
    SampleType makeupGainLin { SampleType (1) };

    void prepareInternal (const juce::dsp::ProcessSpec& specIn) override
    {
        if (specIn.sampleRate <= 0.0 ||
            specIn.maximumBlockSize == 0 ||
            specIn.numChannels == 0)
            return;

        const int maxDelaySamples =
            (int) std::ceil (specIn.sampleRate *
            (maxLookAheadMs * 0.001)) +
            (int) specIn.maximumBlockSize + 2;

        delayBuffer.setSize ((int) specIn.numChannels,
                             maxDelaySamples,
                             false, false, true);

        delayBuffer.clear();

        scFilterStates.assign ((size_t) specIn.numChannels,
                               SampleType (0));

        lookAheadSmoother.reset (specIn.sampleRate, 0.05);
        lookAheadSmoother.setCurrentAndTargetValue (SampleType (0));

        writePos = 0;
        envelopeState = SampleType (0);
        currentGainState = SampleType (1);

        updateFromTargets();
    }

private:

    void updateFromTargets() noexcept
    {
        const double sr = getSampleRate();
        if (sr <= 0.0)
            return;

        thresholdDb =
            (SampleType) thresholdDbTarget.load();

        ratio =
            (SampleType) juce::jmax (1.0,
                ratioTarget.load());

        slope =
            SampleType (1) - SampleType (1) / ratio;

        kneeWidth =
            (SampleType) juce::jmax (0.0,
                kneeWidthTarget.load());

        makeupGainLin =
            juce::Decibels::decibelsToGain (
                (SampleType) makeupDbTarget.load());

        const double atkSamps =
            juce::jmax (0.0001,
                attackMsTarget.load() * 0.001) * sr;

        const double relSamps =
            juce::jmax (0.0001,
                releaseMsTarget.load() * 0.001) * sr;

        attackCoeff =
            (SampleType) (1.0 - std::exp (-1.0 / atkSamps));

        releaseCoeff =
            (SampleType) (1.0 - std::exp (-1.0 / relSamps));

        const double fc =
            juce::jlimit (0.0, sr * 0.49,
                scHpfHzTarget.load());

        scHpfCoeff =
            (SampleType) (1.0 -
                std::exp (-2.0 *
                juce::MathConstants<double>::pi *
                fc / sr));

        const SampleType laSamples =
            (SampleType)
            (juce::jlimit (0.0, maxLookAheadMs,
             lookAheadMsTarget.load()) *
             0.001 * sr);

        lookAheadSmoother.setTargetValue (laSamples);
    }

    static constexpr double maxLookAheadMs = 20.0;

    juce::AudioBuffer<SampleType> delayBuffer;
    juce::SmoothedValue<SampleType,
        juce::ValueSmoothingTypes::Linear> lookAheadSmoother;

    std::vector<SampleType> scFilterStates;

    int writePos { 0 };
    SampleType envelopeState { SampleType (0) };
    SampleType currentGainState { SampleType (1) };

    SampleType attackCoeff  { SampleType (0) };
    SampleType releaseCoeff { SampleType (0) };
    SampleType scHpfCoeff   { SampleType (0) };

    std::atomic<double> thresholdDbTarget { 0.0 };
    std::atomic<double> ratioTarget       { 1.0 };
    std::atomic<double> kneeWidthTarget   { 0.0 };
    std::atomic<double> makeupDbTarget    { 0.0 };
    std::atomic<double> attackMsTarget    { 10.0 };
    std::atomic<double> releaseMsTarget   { 100.0 };
    std::atomic<double> lookAheadMsTarget { 0.0 };
    std::atomic<double> scHpfHzTarget     { 20.0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_DynamicsCore)
};

} // namespace xb
