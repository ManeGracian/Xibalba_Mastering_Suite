#pragma once

#include <JuceHeader.h>
#include <algorithm>
#include <atomic>
#include <cstdint>

#include "XC_DSPBlockBase.h"

namespace xb
{

/*
==============================================================================
    XC_LatencyManager

    Responsabilidad:
    - Calcular latencia total de una cadena activa de DSPBlockBase
    - Definir política de reporte al host:
        * FixedMax: reporta siempre la latencia máxima posible (recomendado)
        * Dynamic: reporta la latencia actual (solo si NO cambias en playback)

    Uso típico:
    - En prepareToPlay(): latencyManager.prepare(spec, maxLatencySamples)
    - Cuando cambie el orden/modos: latencyManager.updateChainLatency(chainLatency)
    - En AudioProcessor:
        setLatencySamples(latencyManager.getReportedLatencySamples());
      y aplicar compensationDelay (delay interno) al audio.

    RT-Safety:
    - No allocs, no locks.
    - Intercambio de datos UI/host->audio vía atomics.
==============================================================================
*/

class XC_LatencyManager final
{
public:
    enum class Policy : int
    {
        FixedMax = 0,
        Dynamic  = 1
    };

    XC_LatencyManager() = default;

    //==========================================================================
    void setPolicy (Policy p) noexcept
    {
        policy.store (p, std::memory_order_release);
        dirty.store (true, std::memory_order_release);
    }

    [[nodiscard]] Policy getPolicy() const noexcept
    {
        return policy.load (std::memory_order_acquire);
    }

    //==========================================================================
    // maxPossibleLatencySamples: la latencia máxima que tu plugin PUEDE tener.
    // Ejemplo: oversampling x8 + lookahead max + etc.
    void prepare (double sampleRate, int maxPossibleLatencySamples) noexcept
    {
        sr = sampleRate;
        maxLatencySamples.store (juce::jmax (0, maxPossibleLatencySamples), std::memory_order_release);

        // Inicialmente asumimos cadena actual = 0
        currentChainLatency.store (0, std::memory_order_release);

        recompute();
    }

    //==========================================================================
    // Se llama cuando cambia la latencia de la cadena activa
    // (por cambio de orden, oversampling mode, lookahead, etc.)
    void updateChainLatency (int chainLatencySamples) noexcept
    {
        currentChainLatency.store (juce::jmax (0, chainLatencySamples), std::memory_order_release);
        dirty.store (true, std::memory_order_release);
    }

    //==========================================================================
    // Llamar en audio thread al inicio de processBlock (barato)
    void tick() noexcept
    {
        if (dirty.exchange (false, std::memory_order_acq_rel))
            recompute();
    }

    //==========================================================================
    // Lo que reportas al host con AudioProcessor::setLatencySamples()
    [[nodiscard]] int getReportedLatencySamples() const noexcept
    {
        return reportedLatency.load (std::memory_order_acquire);
    }

    // Delay interno requerido para igualar la latencia total a la reportada
    // (solo relevante para Policy::FixedMax)
    [[nodiscard]] int getCompensationDelaySamples() const noexcept
    {
        return compensationDelay.load (std::memory_order_acquire);
    }

private:
    void recompute() noexcept
    {
        const int maxLat   = maxLatencySamples.load (std::memory_order_acquire);
        const int chainLat = currentChainLatency.load (std::memory_order_acquire);
        const auto p       = policy.load (std::memory_order_acquire);

        if (p == Policy::Dynamic)
        {
            // Reporta exactamente la latencia actual
            reportedLatency.store (chainLat, std::memory_order_release);
            compensationDelay.store (0, std::memory_order_release);
            return;
        }

        // FixedMax (recomendado): reporta siempre el máximo posible
        reportedLatency.store (maxLat, std::memory_order_release);

        // Compensación interna: lo que falta para llegar al máximo
        const int comp = juce::jmax (0, maxLat - chainLat);
        compensationDelay.store (comp, std::memory_order_release);
    }

    //==========================================================================
    double sr { 44100.0 };

    std::atomic<Policy> policy { Policy::FixedMax };
    std::atomic<bool> dirty { true };

    std::atomic<int> maxLatencySamples { 0 };
    std::atomic<int> currentChainLatency { 0 };

    std::atomic<int> reportedLatency { 0 };
    std::atomic<int> compensationDelay { 0 };
};

} // namespace xb
