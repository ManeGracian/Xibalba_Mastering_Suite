#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>

namespace xb
{

/*
==============================================================================
    XC_SwitchCrossfade

    Crossfade RT-safe para cambios discretos (modo, orden, routing, etc.).
    - No allocs, no locks.
    - Se dispara con trigger().
    - Genera ramp sample-accurate dentro del bloque.

    Uso:
    - prepare(sampleRate, fadeMs)
    - if (paramCambió) crossfade.trigger()
    - en process: crossfade.apply(out, a, b, numChannels, numSamples)

==============================================================================
*/
class XC_SwitchCrossfade final
{
public:
    XC_SwitchCrossfade() = default;

    void prepare (double sampleRate, float fadeMs) noexcept
    {
        sr = (sampleRate > 0.0 ? sampleRate : 44100.0);
        setFadeMs (fadeMs);
        reset();
    }

    void setFadeMs (float fadeMs) noexcept
    {
        const float ms = juce::jlimit (0.0f, 200.0f, fadeMs);
        const int samples = (int) std::ceil ((ms * 0.001f) * (float) sr);
        fadeSamples.store (juce::jmax (1, samples), std::memory_order_release);
    }

    void reset() noexcept
    {
        remaining.store (0, std::memory_order_release);
        phase.store (0, std::memory_order_release);
    }

    // Dispara un crossfade (desde UI/host o audio thread)
    void trigger() noexcept
    {
        remaining.store (fadeSamples.load (std::memory_order_acquire), std::memory_order_release);
        phase.store (0, std::memory_order_release);
    }

    [[nodiscard]] bool isActive() const noexcept
    {
        return remaining.load (std::memory_order_acquire) > 0;
    }

    // Mezcla: out = (1-t)*a + t*b
    // a y b: pointers por canal (float**), no modifica a/b.
    void apply (float** out, const float* const* a, const float* const* b,
                int numChannels, int numSamples) noexcept
    {
        int rem = remaining.load (std::memory_order_acquire);
        if (rem <= 0)
        {
            // out = b (si no hay crossfade activo)
            for (int ch = 0; ch < numChannels; ++ch)
                juce::FloatVectorOperations::copy (out[ch], b[ch], numSamples);
            return;
        }

        const int total = fadeSamples.load (std::memory_order_acquire);
        int ph = phase.load (std::memory_order_acquire);

        for (int i = 0; i < numSamples; ++i)
        {
            const float t = juce::jlimit (0.0f, 1.0f, (float) ph / (float) total);
            const float it = 1.0f - t;

            for (int ch = 0; ch < numChannels; ++ch)
                out[ch][i] = a[ch][i] * it + b[ch][i] * t;

            ++ph;
            if (--rem <= 0)
                break;
        }

        // Si el bloque es más largo que lo que resta, termina copiando el resto desde b
        if (rem <= 0)
        {
            const int done = juce::jmin (numSamples, ph); // ph ya avanzó
            const int left = numSamples - done;

            if (left > 0)
            {
                for (int ch = 0; ch < numChannels; ++ch)
                    juce::FloatVectorOperations::copy (out[ch] + done, b[ch] + done, left);
            }

            remaining.store (0, std::memory_order_release);
            phase.store (0, std::memory_order_release);
        }
        else
        {
            remaining.store (rem, std::memory_order_release);
            phase.store (ph, std::memory_order_release);
        }
    }

private:
    double sr { 44100.0 };
    std::atomic<int> fadeSamples { 256 };
    std::atomic<int> remaining { 0 };
    std::atomic<int> phase { 0 };
};

} // namespace xb
