/*
  ==============================================================================

    XC_LowSculptorEngine.h
    Created: 16 Feb 2026
    Updated: 17 Feb 2026
    Author:  Mane / Xibalba Studios

    Purpose:
    - Engine DSP para X-LowSculptor.
    - Enable/Mute por banda (BODY/NECK) con crossfade anti-click.
    - Gain staging base (IN, BODY, NECK, OUT) RT-safe.
    - Metering IN/OUT (Peak + RMS con ballistics) RT-safe, thread-safe.
    - BODY SQUEEZE (Comp VCA limpio) integrado como primer comp usable.

    Signal Flow:
        IN
          -> IN Gain
          -> IN Meter
          -> Crossover (LR4)
              -> BODY band -> BODY Gain -> BODY SQUEEZE (VCA) -> Enable Fade
              -> NECK band -> NECK Gain -> Enable Fade
          -> SUM (BODY + NECK)
          -> OUT Gain
          -> OUT Meter
          -> OUT

    RT-Safety:
    - No allocs / no locks en process().
    - Buffers dimensionados SOLO en prepare().
    - Atomics para estados (enable).
    - Gains thread-safe via xb::Gain.
    - Meters publican valores via atomics (GUI-safe).

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <atomic>

#include "../../DSP/XC_Crossover.h"
#include "../../DSP/XC_SwitchCrossfade.h"
#include "../../DSP/XC_Gain.h"
#include "../../DSP/XC_Meter.h"

// BODY SQUEEZE (VCA clean comp)
#include "../../DSP/XC_CompVCA.h"

namespace xb
{

class LowSculptorEngine
{
public:
    LowSculptorEngine() = default;

    //==========================================================================
    // Lifecycle
    void prepare (const juce::dsp::ProcessSpec& spec)
    {
        jassert (spec.sampleRate > 0.0);
        jassert (spec.maximumBlockSize > 0);
        jassert (spec.numChannels > 0);

        sampleRate   = spec.sampleRate;
        maxBlockSize = (int) spec.maximumBlockSize;
        numChannels  = (int) spec.numChannels;

        //--------------------------------------------------------------------------
        // Buffers: BODY / NECK
        bodyBuffer.setSize (numChannels, maxBlockSize, false, false, true);
        neckBuffer.setSize (numChannels, maxBlockSize, false, false, true);

        // Prev buffers (para fade-out: prev -> zeros)
        bodyPrev.setSize (numChannels, maxBlockSize, false, false, true);
        neckPrev.setSize (numChannels, maxBlockSize, false, false, true);

        // Zero buffer (fuente para fades)
        zeroBuffer.setSize (numChannels, maxBlockSize, false, false, true);
        zeroBuffer.clear();

        // Punteros por canal (alloc SOLO en prepare)
        outPtrs.allocate ((size_t) numChannels, true); // float**
        aPtrs.allocate   ((size_t) numChannels, true); // const float*
        bPtrs.allocate   ((size_t) numChannels, true); // const float*

        //--------------------------------------------------------------------------
        // DSP blocks
        crossover.prepare (spec);
        crossover.reset();
        crossover.setCutoffHz ((float) defaultCrossoverHz);

        constexpr float fadeMs = 5.0f;
        bodyXfade.prepare (sampleRate, fadeMs);
        neckXfade.prepare (sampleRate, fadeMs);

        // Gain staging
        inGain.prepare   (spec);
        bodyGain.prepare (spec);
        neckGain.prepare (spec);
        outGain.prepare  (spec);

        inGain.setGainDecibels   (0.0f);
        bodyGain.setGainDecibels (0.0f);
        neckGain.setGainDecibels (0.0f);
        outGain.setGainDecibels  (0.0f);

        //--------------------------------------------------------------------------
        // BODY SQUEEZE (VCA clean comp)
        bodySqueeze.prepare (spec);
        bodySqueeze.reset();

        // Defaults musicales y seguros:
        bodySqueeze.setThreshold (-24.0f);
        bodySqueeze.setRatio     (2.0f);
        bodySqueeze.setKnee      (6.0f);
        bodySqueeze.setAttack    (20.0f);
        bodySqueeze.setRelease   (150.0f);
        bodySqueeze.setMakeupGain(0.0f);
        bodySqueeze.setLookAhead (0.0f);
        bodySqueeze.setSidechainHPF (30.0f);

        //--------------------------------------------------------------------------
        // Metering
        inMeter.prepare  (spec);
        outMeter.prepare (spec);

        inMeter.setAttackMs  (5.0f);
        inMeter.setReleaseMs (300.0f);
        outMeter.setAttackMs  (5.0f);
        outMeter.setReleaseMs (300.0f);

        //--------------------------------------------------------------------------
        // Estado inicial: ambas bandas ON
        bodyEnabledTarget.store (true, std::memory_order_release);
        neckEnabledTarget.store (true, std::memory_order_release);
        bodyEnabledCurrent = true;
        neckEnabledCurrent = true;

        prepared = true;
        reset();
    }

    //==========================================================================
    void reset() noexcept
    {
        if (! prepared)
            return;

        crossover.reset();
        bodyXfade.reset();
        neckXfade.reset();

        inGain.reset();
        bodyGain.reset();
        neckGain.reset();
        outGain.reset();

        bodySqueeze.reset();

        inMeter.reset();
        outMeter.reset();

        bodyBuffer.clear();
        neckBuffer.clear();
        bodyPrev.clear();
        neckPrev.clear();
        zeroBuffer.clear();
    }

    //==========================================================================
    void process (juce::AudioBuffer<float>& buffer) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (! prepared)
            return;

        const int chs = buffer.getNumChannels();
        const int n   = buffer.getNumSamples();

        if (chs <= 0 || n <= 0)
            return;

        if (chs > numChannels || n > maxBlockSize)
        {
            jassertfalse;
            return;
        }

        //--------------------------------------------------------------------------
        // 0) IN Gain (pre-crossover)
        inGain.process (buffer);

        //--------------------------------------------------------------------------
        // 1) IN Meter (post-inGain, pre-crossover)
        inMeter.process (buffer);

        //--------------------------------------------------------------------------
        // 2) Split: IN -> BODY/NECK
        {
            juce::dsp::AudioBlock<const float> inBlock (buffer);

            juce::dsp::AudioBlock<float> bodyBlock (bodyBuffer);
            juce::dsp::AudioBlock<float> neckBlock (neckBuffer);

            auto bodySub = bodyBlock.getSubBlock (0, (size_t) n);
            auto neckSub = neckBlock.getSubBlock (0, (size_t) n);

            crossover.process (inBlock, bodySub, neckSub);
        }

        //--------------------------------------------------------------------------
        // 3) Band gains (pre-squeeze / pre-enable fade)
        {
            juce::dsp::AudioBlock<float> bodyBlock (bodyBuffer);
            juce::dsp::AudioBlock<float> neckBlock (neckBuffer);

            auto bodySub = bodyBlock.getSubBlock (0, (size_t) n);
            auto neckSub = neckBlock.getSubBlock (0, (size_t) n);

            bodyGain.process (bodySub);
            neckGain.process (neckSub);
        }

        //--------------------------------------------------------------------------
        // 4) BODY SQUEEZE (VCA) ✅ SOLO BODY
        bodySqueeze.process (bodyBuffer);

        //--------------------------------------------------------------------------
        // 5) Enable/Mute por banda con crossfade anti-click
        applyBandEnable (bodyBuffer, bodyPrev, bodyXfade,
                         bodyEnabledTarget.load (std::memory_order_acquire),
                         bodyEnabledCurrent, chs, n);

        applyBandEnable (neckBuffer, neckPrev, neckXfade,
                         neckEnabledTarget.load (std::memory_order_acquire),
                         neckEnabledCurrent, chs, n);

        //--------------------------------------------------------------------------
        // 6) SUM -> main output buffer
        for (int ch = 0; ch < chs; ++ch)
        {
            auto* out  = buffer.getWritePointer (ch);
            auto* low  = bodyBuffer.getReadPointer (ch);
            auto* high = neckBuffer.getReadPointer (ch);

            juce::FloatVectorOperations::copy (out, low, n);
            juce::FloatVectorOperations::add  (out, high, n);
        }

        //--------------------------------------------------------------------------
        // 7) OUT Gain (post-sum)
        outGain.process (buffer);

        //--------------------------------------------------------------------------
        // 8) OUT Meter (post-outGain)
        outMeter.process (buffer);
    }

    //==========================================================================
    // Public setters (RT-safe)
    void setCrossoverHz (float hz) noexcept { crossover.setCutoffHz (hz); }

    void setInGainDb   (float db) noexcept { inGain.setGainDecibels (db); }
    void setBodyGainDb (float db) noexcept { bodyGain.setGainDecibels (db); }
    void setNeckGainDb (float db) noexcept { neckGain.setGainDecibels (db); }
    void setOutGainDb  (float db) noexcept { outGain.setGainDecibels (db); }

    // ✅ ESTOS ERAN LOS QUE FALTABAN (para que PluginProcessor compile)
    void setBodyEnabled (bool enabled) noexcept
    {
        bodyEnabledTarget.store (enabled, std::memory_order_release);
    }

    void setNeckEnabled (bool enabled) noexcept
    {
        neckEnabledTarget.store (enabled, std::memory_order_release);
    }

    // BODY SQUEEZE (por ahora simple knob: lo mapeas en el processor)
    // Te dejo helper para que tengas “un knob”:
    void setBodySqueezeOneKnob (float amt01) noexcept
    {
        amt01 = juce::jlimit (0.0f, 1.0f, amt01);

        // Mapeo “museo”: suave al inicio, más intenso al final
        const float t = amt01 * amt01;

        // Threshold baja con amount
        const float thr = juce::jmap (t, -18.0f, -38.0f);

        // Ratio sube con amount
        const float ratio = juce::jmap (t, 1.2f, 4.0f);

        // Knee mantiene musicalidad
        const float knee = juce::jmap (amt01, 9.0f, 3.0f);

        // Attack/Release estilo “glue”
        const float atk = juce::jmap (amt01, 35.0f, 10.0f);
        const float rel = juce::jmap (amt01, 250.0f, 120.0f);

        bodySqueeze.setThreshold (thr);
        bodySqueeze.setRatio (ratio);
        bodySqueeze.setKnee (knee);
        bodySqueeze.setAttack (atk);
        bodySqueeze.setRelease (rel);

        // Makeup 0 por defecto (mantenlo limpio)
        bodySqueeze.setMakeupGain (0.0f);
    }

    //==========================================================================
    // Meter getters (GUI-safe)
    [[nodiscard]] float getInPeakDb()  const noexcept { return inMeter.getPeakDb(); }
    [[nodiscard]] float getInRmsDb()   const noexcept { return inMeter.getRmsDb();  }
    [[nodiscard]] float getOutPeakDb() const noexcept { return outMeter.getPeakDb(); }
    [[nodiscard]] float getOutRmsDb()  const noexcept { return outMeter.getRmsDb();  }

private:
    //==========================================================================
    void applyBandEnable (juce::AudioBuffer<float>& band,
                          juce::AudioBuffer<float>& bandPrevBuffer,
                          xb::XC_SwitchCrossfade& xfade,
                          bool targetEnabled,
                          bool& currentEnabled,
                          int chs,
                          int n) noexcept
    {
        if (targetEnabled != currentEnabled)
        {
            if (! targetEnabled)
            {
                for (int ch = 0; ch < chs; ++ch)
                {
                    juce::FloatVectorOperations::copy (bandPrevBuffer.getWritePointer (ch),
                                                       band.getReadPointer (ch),
                                                       n);
                }
            }

            xfade.trigger();
            currentEnabled = targetEnabled;
        }

        if (! xfade.isActive())
        {
            if (! currentEnabled)
                band.clear();

            return;
        }

        for (int ch = 0; ch < chs; ++ch)
        {
            outPtrs[ch] = band.getWritePointer (ch);

            if (currentEnabled)
            {
                aPtrs[ch] = zeroBuffer.getReadPointer (ch);
                bPtrs[ch] = band.getReadPointer (ch);
            }
            else
            {
                aPtrs[ch] = bandPrevBuffer.getReadPointer (ch);
                bPtrs[ch] = zeroBuffer.getReadPointer (ch);
            }
        }

        xfade.apply (outPtrs.get(), aPtrs.get(), bPtrs.get(), chs, n);
    }

    //==========================================================================
    static constexpr double defaultCrossoverHz = 150.0;

    bool prepared { false };

    double sampleRate { 44100.0 };
    int maxBlockSize  { 512 };
    int numChannels   { 2 };

    // DSP blocks
    xb::Crossover<float> crossover;

    // Gain staging
    xb::Gain<float> inGain;
    xb::Gain<float> bodyGain;
    xb::Gain<float> neckGain;
    xb::Gain<float> outGain;

    // BODY SQUEEZE (VCA)
    xb::XC_CompVCA<float> bodySqueeze;

    // Metering
    xb::Meter<float> inMeter;
    xb::Meter<float> outMeter;

    // Band buffers
    juce::AudioBuffer<float> bodyBuffer, neckBuffer;
    juce::AudioBuffer<float> bodyPrev, neckPrev;
    juce::AudioBuffer<float> zeroBuffer;

    // Band enable crossfades
    xb::XC_SwitchCrossfade bodyXfade;
    xb::XC_SwitchCrossfade neckXfade;

    // Enable state (UI/host -> audio)
    std::atomic<bool> bodyEnabledTarget { true };
    std::atomic<bool> neckEnabledTarget { true };

    // Audio-thread stable state
    bool bodyEnabledCurrent { true };
    bool neckEnabledCurrent { true };

    // Punteros por canal (alloc solo en prepare)
    juce::HeapBlock<float*>       outPtrs;
    juce::HeapBlock<const float*> aPtrs;
    juce::HeapBlock<const float*> bPtrs;
};

} // namespace xb

