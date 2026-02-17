#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <array>
#include <vector>

#include "XC_DSPBlockBase.h"
#include "XC_DelayAligner.h"
#include "XC_SwitchCrossfade.h"

namespace xb
{

class XC_Router final : public DSPBlockBase
{
public:
    enum class Mode : int
    {
        Serial   = 0,
        Parallel = 1, // Dry/Wet
        Multiband = 2 // 3-way
    };

    static constexpr int kMaxBlocksPerChain = 16;

    XC_Router() = default;
    ~XC_Router() override = default;

    //==========================================================================
    // Configuración (UI/host thread friendly)
    void setMode (Mode m) noexcept
    {
        requestedMode.store (m, std::memory_order_release);
        dirty.store (true, std::memory_order_release);
    }

    void setCrossfadeMs (float ms) noexcept
    {
        xfadeMs.store (juce::jlimit (0.0f, 200.0f, ms), std::memory_order_release);
        dirty.store (true, std::memory_order_release);
    }

    // Parallel dry/wet
    void setMix01 (float mix01) noexcept
    {
        mix.store (juce::jlimit (0.0f, 1.0f, mix01), std::memory_order_release);
    }

    // Multiband cutoffs
    void setMultibandCutoffs (float lowMidHz, float midHighHz) noexcept
    {
        mbLowMidHz.store (juce::jlimit (20.0f, 20000.0f, lowMidHz), std::memory_order_release);
        mbMidHighHz.store (juce::jlimit (20.0f, 20000.0f, midHighHz), std::memory_order_release);
        dirty.store (true, std::memory_order_release);
    }

    // Multiband gains
    void setBandGains (float low, float mid, float high) noexcept
    {
        bandGainL.store (low, std::memory_order_release);
        bandGainM.store (mid, std::memory_order_release);
        bandGainH.store (high, std::memory_order_release);
    }

    //==========================================================================
    // Cadenas

    void setSerialChain (const std::array<DSPBlockBase*, kMaxBlocksPerChain>& blocks, int count) noexcept
    {
        serialBlocks = blocks;
        serialCount  = juce::jlimit (0, kMaxBlocksPerChain, count);
        dirty.store (true, std::memory_order_release);
    }

    // Parallel: rama Wet (la Dry es “passthrough”)
    void setWetChain (const std::array<DSPBlockBase*, kMaxBlocksPerChain>& blocks, int count) noexcept
    {
        wetBlocks = blocks;
        wetCount  = juce::jlimit (0, kMaxBlocksPerChain, count);
        dirty.store (true, std::memory_order_release);
    }

    // Multiband: cadenas por banda
    void setBandChains (const std::array<DSPBlockBase*, kMaxBlocksPerChain>& low,  int lowCount,
                        const std::array<DSPBlockBase*, kMaxBlocksPerChain>& mid,  int midCount,
                        const std::array<DSPBlockBase*, kMaxBlocksPerChain>& high, int highCount) noexcept
    {
        mbBlocksL = low;  mbCountL = juce::jlimit (0, kMaxBlocksPerChain, lowCount);
        mbBlocksM = mid;  mbCountM = juce::jlimit (0, kMaxBlocksPerChain, midCount);
        mbBlocksH = high; mbCountH = juce::jlimit (0, kMaxBlocksPerChain, highCount);
        dirty.store (true, std::memory_order_release);
    }

    //==========================================================================
    void reset() noexcept override
    {
        delayWet.reset();
        delayL.reset();
        delayM.reset();
        delayH.reset();

        xfadeAlignA.reset();
        xfadeAlignB.reset();
        xfade.reset();

        for (auto& f : lpLowMid)  f.reset();
        for (auto& f : hpLowMid)  f.reset();
        for (auto& f : lpMidHigh) f.reset();
        for (auto& f : hpMidHigh) f.reset();
    }

    //==========================================================================
    void process (juce::AudioBuffer<float>& buffer) noexcept
    {
        juce::dsp::AudioBlock<float> block (buffer);
        process (block);
    }

    void process (juce::dsp::AudioBlock<float>& block) noexcept
    {
        juce::ScopedNoDenormals noDenormals;

        if (isBypassed())
            return;

        const int chs = (int) block.getNumChannels();
        const int n   = (int) block.getNumSamples();
        if (chs <= 0 || n <= 0)
            return;

        applyIfDirty(); // aplica cambios discretos

        // Render normal: si no hay crossfade activo -> render modo activo directo
        if (!xfade.isActive())
        {
            renderModeInto (block, renderB, activeMode);
            copyBufferToBlock (renderB, block, n);
            return;
        }

        // Crossfade activo: render ambos modos
        renderModeInto (block, renderA, fromMode);
        renderModeInto (block, renderB, toMode);

        // Alinear outputs (por si latencias globales difieren)
        alignForCrossfade (renderA, renderB, n);

        // Mezclar sample-accurate
        auto out = makePtrArray (block);
        auto a   = makePtrArrayConst (renderA, chs);
        auto b   = makePtrArrayConst (renderB, chs);

        xfade.apply (out.data(), a.data(), b.data(), chs, n);
    }

    //==========================================================================
    [[nodiscard]] int getLatencySamples() const noexcept override
    {
        return activeLatency.load (std::memory_order_acquire);
    }

    // Para política FixedMax del plugin: el máximo posible (según tus cadenas + settings).
    // Esta versión es conservadora: toma máximo entre modos “posibles” con la config actual.
    [[nodiscard]] int getMaxLatencySamples() const noexcept
    {
        return maxLatencyPossible.load (std::memory_order_acquire);
    }

protected:
    //==========================================================================
    void prepareInternal (const juce::dsp::ProcessSpec& spec) override
    {
        if (spec.sampleRate <= 0.0 || spec.maximumBlockSize == 0 || spec.numChannels == 0)
            return;

        sampleRate = spec.sampleRate;
        maxBlock   = (int) spec.maximumBlockSize;
        channels   = (int) spec.numChannels;

        // Buffers de render
        renderA.setSize (channels, maxBlock, false, false, true);
        renderB.setSize (channels, maxBlock, false, false, true);

        // Buffers internos para parallel/MB
        dryBuf.setSize (channels, maxBlock, false, false, true);
        wetBuf.setSize (channels, maxBlock, false, false, true);

        bandL.setSize (channels, maxBlock, false, false, true);
        bandM.setSize (channels, maxBlock, false, false, true);
        bandH.setSize (channels, maxBlock, false, false, true);

        // Delays para alineación interna
        const int maxDelay = 8192;
        delayWet.setMaxDelaySamples (maxDelay);
        delayL.setMaxDelaySamples   (maxDelay);
        delayM.setMaxDelaySamples   (maxDelay);
        delayH.setMaxDelaySamples   (maxDelay);

        xfadeAlignA.setMaxDelaySamples (maxDelay);
        xfadeAlignB.setMaxDelaySamples (maxDelay);

        delayWet.prepare (spec);
        delayL.prepare (spec);
        delayM.prepare (spec);
        delayH.prepare (spec);

        xfadeAlignA.prepare (spec);
        xfadeAlignB.prepare (spec);

        // Filtros LR por canal (MB)
        lpLowMid.assign  (channels, juce::dsp::LinkwitzRileyFilter<float>());
        hpLowMid.assign  (channels, juce::dsp::LinkwitzRileyFilter<float>());
        lpMidHigh.assign (channels, juce::dsp::LinkwitzRileyFilter<float>());
        hpMidHigh.assign (channels, juce::dsp::LinkwitzRileyFilter<float>());

        dirty.store (true, std::memory_order_release);
        applyIfDirty();
        reset();
    }

private:
    //==========================================================================
    static std::array<float*, 64> makePtrArray (juce::dsp::AudioBlock<float>& block) noexcept
    {
        std::array<float*, 64> ptrs {};
        const int chs = (int) block.getNumChannels();
        for (int ch = 0; ch < chs; ++ch)
            ptrs[(size_t) ch] = block.getChannelPointer ((size_t) ch);
        return ptrs;
    }

    static std::array<const float*, 64> makePtrArrayConst (const juce::AudioBuffer<float>& buf, int chs) noexcept
    {
        std::array<const float*, 64> ptrs {};
        for (int ch = 0; ch < chs; ++ch)
            ptrs[(size_t) ch] = buf.getReadPointer (ch);
        return ptrs;
    }

    //==========================================================================
    static void copyBlockToBuffer (juce::dsp::AudioBlock<float>& src,
                                  juce::AudioBuffer<float>& dst,
                                  int numSamples) noexcept
    {
        const int chs = (int) src.getNumChannels();
        for (int ch = 0; ch < chs; ++ch)
            juce::FloatVectorOperations::copy (dst.getWritePointer (ch), src.getChannelPointer ((size_t) ch), numSamples);
    }

    static void copyBufferToBlock (const juce::AudioBuffer<float>& src,
                                  juce::dsp::AudioBlock<float>& dst,
                                  int numSamples) noexcept
    {
        const int chs = (int) dst.getNumChannels();
        for (int ch = 0; ch < chs; ++ch)
            juce::FloatVectorOperations::copy (dst.getChannelPointer ((size_t) ch), src.getReadPointer (ch), numSamples);
    }

    static void clearBuffer (juce::AudioBuffer<float>& buf, int chs, int n) noexcept
    {
        for (int ch = 0; ch < chs; ++ch)
            juce::FloatVectorOperations::clear (buf.getWritePointer (ch), n);
    }

    //==========================================================================
    static int chainLatencySum (const std::array<DSPBlockBase*, kMaxBlocksPerChain>& blocks, int count) noexcept
    {
        int sum = 0;
        for (int i = 0; i < count; ++i)
            if (blocks[i] != nullptr)
                sum += blocks[i]->getLatencySamples();
        return sum;
    }

    static void processChain (juce::dsp::AudioBlock<float>& block,
                              const std::array<DSPBlockBase*, kMaxBlocksPerChain>& blocks,
                              int count) noexcept
    {
        for (int i = 0; i < count; ++i)
            if (auto* b = blocks[i])
                b->process (block);
    }

    //==========================================================================
    void applyIfDirty() noexcept
    {
        if (!dirty.exchange (false, std::memory_order_acq_rel))
            return;

        const Mode newMode = requestedMode.load (std::memory_order_acquire);

        // Cutoffs
        const float f1 = mbLowMidHz.load (std::memory_order_acquire);
        const float f2 = mbMidHighHz.load (std::memory_order_acquire);

        for (int ch = 0; ch < channels; ++ch)
        {
            lpLowMid [ch].setCutoffFrequency  (sampleRate, f1);
            hpLowMid [ch].setCutoffFrequency  (sampleRate, f1);
            lpMidHigh[ch].setCutoffFrequency  (sampleRate, f2);
            hpMidHigh[ch].setCutoffFrequency  (sampleRate, f2);

            lpLowMid [ch].setType (juce::dsp::LinkwitzRileyFilterType::lowpass);
            hpLowMid [ch].setType (juce::dsp::LinkwitzRileyFilterType::highpass);
            lpMidHigh[ch].setType (juce::dsp::LinkwitzRileyFilterType::lowpass);
            hpMidHigh[ch].setType (juce::dsp::LinkwitzRileyFilterType::highpass);
        }

        // Crossfade config
        const float ms = xfadeMs.load (std::memory_order_acquire);
        xfade.prepare (sampleRate, ms);

        // Recalcular latencias por modo (con config actual)
        const int latSerial = chainLatencySum (serialBlocks, serialCount);

        // Parallel: Dry=0 latency, Wet=latWet; global=max(0, latWet)=latWet
        const int latWet = chainLatencySum (wetBlocks, wetCount);
        const int latParallel = latWet;

        // Multiband: max de bandas (cada banda suma su cadena)
        const int latL = chainLatencySum (mbBlocksL, mbCountL);
        const int latM = chainLatencySum (mbBlocksM, mbCountM);
        const int latH = chainLatencySum (mbBlocksH, mbCountH);
        const int latMB = juce::jmax (latL, juce::jmax (latM, latH));

        // Máximo posible (con esta configuración actual)
        const int maxLat = juce::jmax (latSerial, juce::jmax (latParallel, latMB));
        maxLatencyPossible.store (maxLat, std::memory_order_release);

        // Configurar delays internos de alineación para parallel/MB
        // Parallel: alinear Dry hacia Wet
        delayWet.setDelaySamples (0); // Wet es la referencia (máxima)
        // Dry delay lo aplicamos usando xfadeAlign o delay externo? aquí: delayWet solo es para Wet (no hace falta).
        // Para Dry usaremos un delay "implícito" aplicando delay a dryBuf si latWet > 0.
        dryDelaySamples = juce::jmax (0, latWet); // Dry debe retrasarse latWet para alinearse con Wet.

        // MB: alinear bandas al máximo
        delayL.setDelaySamples (latMB - latL);
        delayM.setDelaySamples (latMB - latM);
        delayH.setDelaySamples (latMB - latH);

        // Si cambia el modo activo => dispara crossfade
        if (newMode != activeMode)
        {
            fromMode = activeMode;
            toMode   = newMode;
            activeMode = newMode;

            // Durante el fade, latencia efectiva debe ser max(ambas) para evitar desfases
            const int latFrom = latencyForMode (fromMode, latSerial, latParallel, latMB);
            const int latTo   = latencyForMode (toMode,   latSerial, latParallel, latMB);
            const int latX    = juce::jmax (latFrom, latTo);

            activeLatency.store (latX, std::memory_order_release);

            // Programar alineación extra para crossfade (para que A y B queden con misma latencia)
            xfadeAlignA.setDelaySamples (latX - latFrom);
            xfadeAlignB.setDelaySamples (latX - latTo);

            xfade.trigger();
        }
        else
        {
            // Sin cambio de modo: latencia activa normal
            const int lat = latencyForMode (activeMode, latSerial, latParallel, latMB);
            activeLatency.store (lat, std::memory_order_release);
        }
    }

    static int latencyForMode (Mode m, int latSerial, int latParallel, int latMB) noexcept
    {
        switch (m)
        {
            case Mode::Serial:   return latSerial;
            case Mode::Parallel: return latParallel;
            case Mode::Multiband:return latMB;
            default:             return latSerial;
        }
    }

    //==========================================================================
    void renderModeInto (juce::dsp::AudioBlock<float>& input,
                         juce::AudioBuffer<float>& out,
                         Mode mode) noexcept
    {
        const int chs = (int) input.getNumChannels();
        const int n   = (int) input.getNumSamples();

        // Start from input
        copyBlockToBuffer (input, out, n);
        juce::dsp::AudioBlock<float> blockOut (out);
        blockOut = blockOut.getSubBlock (0, (size_t) n);

        if (mode == Mode::Serial)
        {
            processChain (blockOut, serialBlocks, serialCount);
            return;
        }

        if (mode == Mode::Parallel)
        {
            renderParallel (input, out, n);
            return;
        }

        renderMultiband (input, out, n);
    }

    //==========================================================================
    void renderParallel (juce::dsp::AudioBlock<float>& input,
                         juce::AudioBuffer<float>& out,
                         int n) noexcept
    {
        const int chs = (int) input.getNumChannels();

        // Dry/Wet buffers
        copyBlockToBuffer (input, dryBuf, n);
        copyBlockToBuffer (input, wetBuf, n);

        auto dry = juce::dsp::AudioBlock<float> (dryBuf).getSubBlock (0, (size_t) n);
        auto wet = juce::dsp::AudioBlock<float> (wetBuf).getSubBlock (0, (size_t) n);

        // Procesar wet chain
        processChain (wet, wetBlocks, wetCount);

        // Alinear dry a wet (si latWet > 0)
        if (dryDelaySamples > 0)
        {
            // Reutilizamos xfadeAlignA como delay simple para dry (sin crossfade)
            xfadeAlignA.setDelaySamples (dryDelaySamples);
            xfadeAlignA.process (dry);
            xfadeAlignA.setDelaySamples (0);
        }

        // Mix
        const float m = mix.load (std::memory_order_acquire);
        const float gDry = 1.0f - m;
        const float gWet = m;

        clearBuffer (out, chs, n);
        for (int ch = 0; ch < chs; ++ch)
        {
            auto* dst = out.getWritePointer (ch);
            juce::FloatVectorOperations::addWithMultiply (dst, dry.getChannelPointer ((size_t) ch), gDry, n);
            juce::FloatVectorOperations::addWithMultiply (dst, wet.getChannelPointer ((size_t) ch), gWet, n);
        }
    }

    //==========================================================================
    void renderMultiband (juce::dsp::AudioBlock<float>& input,
                          juce::AudioBuffer<float>& out,
                          int n) noexcept
    {
        const int chs = (int) input.getNumChannels();

        // Copiar input a bandas
        copyBlockToBuffer (input, bandL, n);
        copyBlockToBuffer (input, bandM, n);
        copyBlockToBuffer (input, bandH, n);

        auto bL = juce::dsp::AudioBlock<float> (bandL).getSubBlock (0, (size_t) n);
        auto bM = juce::dsp::AudioBlock<float> (bandM).getSubBlock (0, (size_t) n);
        auto bH = juce::dsp::AudioBlock<float> (bandH).getSubBlock (0, (size_t) n);

        // Split LR (por sample, por canal)
        for (int ch = 0; ch < chs; ++ch)
        {
            auto* l = bL.getChannelPointer ((size_t) ch);
            auto* m = bM.getChannelPointer ((size_t) ch);
            auto* h = bH.getChannelPointer ((size_t) ch);

            auto& lp1 = lpLowMid[ch];
            auto& hp1 = hpLowMid[ch];
            auto& lp2 = lpMidHigh[ch];
            auto& hp2 = hpMidHigh[ch];

            for (int i = 0; i < n; ++i)
            {
                const float x = l[i];

                const float low  = lp1.processSample (x);
                const float high = hp2.processSample (x);
                const float mid  = lp2.processSample (hp1.processSample (x));

                l[i] = low;
                m[i] = mid;
                h[i] = high;
            }
        }

        // Procesar cadenas por banda
        processChain (bL, mbBlocksL, mbCountL);
        processChain (bM, mbBlocksM, mbCountM);
        processChain (bH, mbBlocksH, mbCountH);

        // Alinear por banda
        delayL.process (bL);
        delayM.process (bM);
        delayH.process (bH);

        // Gains por banda
        const float gL = bandGainL.load (std::memory_order_acquire);
        const float gM = bandGainM.load (std::memory_order_acquire);
        const float gH = bandGainH.load (std::memory_order_acquire);

        // Sumar a out
        clearBuffer (out, chs, n);

        for (int ch = 0; ch < chs; ++ch)
        {
            auto* dst = out.getWritePointer (ch);
            juce::FloatVectorOperations::addWithMultiply (dst, bL.getChannelPointer ((size_t) ch), gL, n);
            juce::FloatVectorOperations::addWithMultiply (dst, bM.getChannelPointer ((size_t) ch), gM, n);
            juce::FloatVectorOperations::addWithMultiply (dst, bH.getChannelPointer ((size_t) ch), gH, n);
        }
    }

    //==========================================================================
    void alignForCrossfade (juce::AudioBuffer<float>& a,
                            juce::AudioBuffer<float>& b,
                            int n) noexcept
    {
        // Aplica delays extra calculados en applyIfDirty()
        auto aBlk = juce::dsp::AudioBlock<float> (a).getSubBlock (0, (size_t) n);
        auto bBlk = juce::dsp::AudioBlock<float> (b).getSubBlock (0, (size_t) n);

        xfadeAlignA.process (aBlk);
        xfadeAlignB.process (bBlk);

        // reset a 0 para que no afecte otros usos si no hay fade, pero es seguro dejarlo.
        // (aquí lo dejamos, porque solo se setea en cambios de modo)
    }

    //==========================================================================
    // Estado preparado
    double sampleRate { 44100.0 };
    int maxBlock { 512 };
    int channels { 2 };

    // Modo + crossfade
    std::atomic<Mode> requestedMode { Mode::Serial };
    Mode activeMode { Mode::Serial };
    Mode fromMode { Mode::Serial };
    Mode toMode   { Mode::Serial };

    std::atomic<float> xfadeMs { 12.0f };
    std::atomic<bool> dirty { true };
    XC_SwitchCrossfade xfade;

    // Latencias
    std::atomic<int> activeLatency { 0 };
    std::atomic<int> maxLatencyPossible { 0 };

    // Parallel: mix
    std::atomic<float> mix { 0.0f };
    int dryDelaySamples { 0 }; // para alinear dry con wet (latWet)

    // Multiband params
    std::atomic<float> mbLowMidHz  { 200.0f };
    std::atomic<float> mbMidHighHz { 2000.0f };
    std::atomic<float> bandGainL { 1.0f };
    std::atomic<float> bandGainM { 1.0f };
    std::atomic<float> bandGainH { 1.0f };

    // Chains
    std::array<DSPBlockBase*, kMaxBlocksPerChain> serialBlocks {};
    int serialCount { 0 };

    std::array<DSPBlockBase*, kMaxBlocksPerChain> wetBlocks {};
    int wetCount { 0 };

    std::array<DSPBlockBase*, kMaxBlocksPerChain> mbBlocksL {}, mbBlocksM {}, mbBlocksH {};
    int mbCountL { 0 }, mbCountM { 0 }, mbCountH { 0 };

    // Delays internos de alineación
    XC_DelayAligner<float> delayWet; // (reservado si luego quieres alinear wet también)
    XC_DelayAligner<float> delayL, delayM, delayH;

    // Delays extra para crossfade (alineación entre modos)
    XC_DelayAligner<float> xfadeAlignA, xfadeAlignB;

    // Buffers internos
    juce::AudioBuffer<float> renderA, renderB;
    juce::AudioBuffer<float> dryBuf, wetBuf;
    juce::AudioBuffer<float> bandL, bandM, bandH;

    // Filtros LR por canal
    std::vector<juce::dsp::LinkwitzRileyFilter<float>> lpLowMid, hpLowMid, lpMidHigh, hpMidHigh;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XC_Router)
};

} // namespace xb
