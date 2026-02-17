/*
  ==============================================================================

    XC_LowSculptorEngine.h
    Created: 16 Feb 2026
    Updated: 16 Feb 2026 (Paso 7 - Gain staging + PluginDoctor tests)
    Author:  Mane / Xibalba Studios

    Purpose:
    - Engine DSP para X-LowSculptor.
    - Paso 6: Enable/Mute por banda (BODY/NECK) con crossfade anti-click.
    - Paso 7: Gain staging base (IN, BODY, NECK, OUT) RT-safe.
    - Incluye modo TEST (sin UI) para validar en PluginDoctor.

    Signal Flow:
        IN
          -> IN Gain
          -> Crossover (LR4)
              -> BODY band -> BODY Gain -> (Enable Fade)
              -> NECK band -> NECK Gain -> (Enable Fade)
          -> SUM (BODY + NECK)
          -> OUT Gain
          -> OUT

    RT-Safety:
    - No allocs / no locks en process().
    - Buffers + punteros dimensionados SOLO en prepare().
    - Atomics para estados de enable.
    - Gains thread-safe via xb::Gain (atomics + dirty flags).

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <atomic>

#include "../../DSP/XC_Crossover.h"
#include "../../DSP/XC_SwitchCrossfade.h"
#include "../../DSP/XC_Gain.h"

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

        // Gain staging (Paso 7)
        inGain.prepare   (spec);
        bodyGain.prepare (spec);
        neckGain.prepare (spec);
        outGain.prepare  (spec);

        //--------------------------------------------------------------------------
        // Defaults (unity)
        inGain.setGainDecibels   (0.0f);
        bodyGain.setGainDecibels (0.0f);
        neckGain.setGainDecibels (0.0f);
        outGain.setGainDecibels  (0.0f);

        //--------------------------------------------------------------------------
        // TEST MODE (PluginDoctor) - SIN UI
        // Cambia SOLO esta constante para hacer pruebas.
        applyPluginDoctorTestMode();

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
        // 1) Split: IN -> BODY/NECK
        {
            juce::dsp::AudioBlock<const float> inBlock (buffer);

            juce::dsp::AudioBlock<float> bodyBlock (bodyBuffer);
            juce::dsp::AudioBlock<float> neckBlock (neckBuffer);

            auto bodySub = bodyBlock.getSubBlock (0, (size_t) n);
            auto neckSub = neckBlock.getSubBlock (0, (size_t) n);

            crossover.process (inBlock, bodySub, neckSub);
        }

        //--------------------------------------------------------------------------
        // 2) Band gains (pre-enable fade)
        // Nota: Gain::process(AudioBlock&) toma ref no-const, así que guardamos subblocks.
        {
            juce::dsp::AudioBlock<float> bodyBlock (bodyBuffer);
            juce::dsp::AudioBlock<float> neckBlock (neckBuffer);

            auto bodySub = bodyBlock.getSubBlock (0, (size_t) n);
            auto neckSub = neckBlock.getSubBlock (0, (size_t) n);

            bodyGain.process (bodySub);
            neckGain.process (neckSub);
        }

        //--------------------------------------------------------------------------
        // 3) Enable/Mute por banda con crossfade anti-click
        applyBandEnable (bodyBuffer, bodyPrev, bodyXfade,
                         bodyEnabledTarget.load (std::memory_order_acquire),
                         bodyEnabledCurrent, chs, n);

        applyBandEnable (neckBuffer, neckPrev, neckXfade,
                         neckEnabledTarget.load (std::memory_order_acquire),
                         neckEnabledCurrent, chs, n);

        //--------------------------------------------------------------------------
        // 4) SUM -> main output buffer
        for (int ch = 0; ch < chs; ++ch)
        {
            auto* out  = buffer.getWritePointer (ch);
            auto* low  = bodyBuffer.getReadPointer (ch);
            auto* high = neckBuffer.getReadPointer (ch);

            juce::FloatVectorOperations::copy (out, low, n);
            juce::FloatVectorOperations::add  (out, high, n);
        }

        //--------------------------------------------------------------------------
        // 5) OUT Gain (post-sum)
        outGain.process (buffer);
    }

    //==========================================================================
    // Public setters (RT-safe)
    void setCrossoverHz (float hz) noexcept
    {
        crossover.setCutoffHz (hz);
    }

    void setInGainDb (float db) noexcept
    {
        inGain.setGainDecibels (db);
    }

    void setBodyGainDb (float db) noexcept
    {
        bodyGain.setGainDecibels (db);
    }

    void setNeckGainDb (float db) noexcept
    {
        neckGain.setGainDecibels (db);
    }

    void setOutGainDb (float db) noexcept
    {
        outGain.setGainDecibels (db);
    }

    void setBodyEnabled (bool enabled) noexcept
    {
        bodyEnabledTarget.store (enabled, std::memory_order_release);
    }

    void setNeckEnabled (bool enabled) noexcept
    {
        neckEnabledTarget.store (enabled, std::memory_order_release);
    }

private:
    //==========================================================================
    // PluginDoctor test mode selector (SIN UI)
    enum class TestMode
    {
        Off = 0,

        // IN gain tests
        InPlus6,
        InMinus6,

        // OUT gain tests
        OutPlus6,
        OutMinus6,

        // Band solo tests (via gains)
        SoloBody,   // NECK muy bajo
        SoloNeck    // BODY muy bajo
    };

    //*** 🔴 CAMBIA AQUÍ para probar una por una en PluginDoctor:
    static constexpr TestMode kTestMode = TestMode::Off;

    void applyPluginDoctorTestMode() noexcept
    {
        // Defaults unity ya aplicados antes de llamar aquí.
        // Aquí solo sobrescribimos según el modo.

        switch (kTestMode)
        {
            case TestMode::Off:
                // no-op
                break;

            case TestMode::InPlus6:
                inGain.setGainDecibels (6.0f);
                break;

            case TestMode::InMinus6:
                inGain.setGainDecibels (-6.0f);
                break;

            case TestMode::OutPlus6:
                outGain.setGainDecibels (6.0f);
                break;

            case TestMode::OutMinus6:
                outGain.setGainDecibels (-6.0f);
                break;

            case TestMode::SoloBody:
                // NECK casi apagado
                neckGain.setGainDecibels (-60.0f);
                break;

            case TestMode::SoloNeck:
                // BODY casi apagado
                bodyGain.setGainDecibels (-60.0f);
                break;
        }
    }

    //==========================================================================
    // Band enable crossfade helper (RT-safe)
    void applyBandEnable (juce::AudioBuffer<float>& band,
                          juce::AudioBuffer<float>& bandPrevBuffer,
                          xb::XC_SwitchCrossfade& xfade,
                          bool targetEnabled,
                          bool& currentEnabled,
                          int chs,
                          int n) noexcept
    {
        // Detecta cambio -> dispara crossfade
        if (targetEnabled != currentEnabled)
        {
            // Si vamos a APAGAR: congelamos audio actual como fuente A (prev)
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

        // Sin crossfade activo -> estado estable
        if (! xfade.isActive())
        {
            if (! currentEnabled)
                band.clear();

            return;
        }

        // Crossfade activo:
        // ON  : out = (1-t)*zeros + t*band  (fade-in)
        // OFF : out = (1-t)*prev  + t*zeros (fade-out)
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

    // Gain staging (Paso 7)
    xb::Gain<float> inGain;
    xb::Gain<float> bodyGain;
    xb::Gain<float> neckGain;
    xb::Gain<float> outGain;

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
