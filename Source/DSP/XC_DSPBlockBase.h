#pragma once
#include <JuceHeader.h>

namespace xb
{

/*
==============================================================================
    xb::DSPBlockBase

    Base abstracta para bloques DSP en Xibalbá DSP (JUCE 8).

    Objetivos de diseño:
    - Real-Time Safe por contrato: nada de asignaciones / locks en process().
    - Channel-agnostic: el bloque debe funcionar con N canales (1/2 típicamente),
      sin suposiciones hardcoded.
    - Inicialización controlada (NVI): prepare() NO virtual, delega a prepareInternal().
    - Compatible con JUCE dsp::ProcessSpec.
    - Sin virtual en la ruta de audio: process() se resuelve por duck typing /
      templates en el contenedor/chain (si mantienes ese enfoque).

    Nota:
    - Este base NO define process() a propósito (zero-overhead en ruta de audio).
      Si necesitas ruteo dinámico (runtime) con reordenamiento por UI, normalmente
      te conviene una segunda interfaz con process() virtual. Podemos añadirla
      después sin romper esta base.
==============================================================================
*/

class DSPBlockBase
{
public:
    DSPBlockBase() = default;
    virtual ~DSPBlockBase() = default;

    //==========================================================================
    // Preparación (NVI) — Preferida: JUCE dsp::ProcessSpec
    void prepare (const juce::dsp::ProcessSpec& spec)
    {
        // Validación robusta contra valores basura en scan/instanciación.
        const auto safe = makeSafeSpec (spec);

        sampleRate       = safe.sampleRate;
        maximumBlockSize = static_cast<int> (safe.maximumBlockSize);
        numChannels      = static_cast<int> (safe.numChannels);
        isPreparedFlag   = true;

        // Hook a la clase derivada (asignaciones/coeficientes SOLO aquí).
        prepareInternal (safe);
    }

    // Backward/Convenience overload (por si tu código ya llama con valores sueltos)
    void prepare (double newSampleRate, int newMaximumBlockSize, int newNumChannels)
    {
        juce::dsp::ProcessSpec spec;
        spec.sampleRate       = newSampleRate;
        spec.maximumBlockSize = static_cast<juce::uint32> (juce::jmax (0, newMaximumBlockSize));
        spec.numChannels      = static_cast<juce::uint32> (juce::jmax (0, newNumChannels));
        prepare (spec);
    }

    //==========================================================================
    // Reset del estado DSP (delays, integradores, envelopes, etc.)
    // IMPORTANTE: debe ser RT-safe (sin allocs).
    virtual void reset() noexcept = 0;

    //==========================================================================
    // Bypass “por contrato” (opcional, pero útil para chains/grafos)
    void setBypassed (bool shouldBypass) noexcept     { bypassedFlag = shouldBypass; }
    [[nodiscard]] bool isBypassed() const noexcept    { return bypassedFlag; }

    //==========================================================================
    // Latencia por bloque (si aplica: oversampling, lookahead, FIR, etc.)
    // Por defecto: 0. Si un bloque introduce latencia, sobreescribe esto.
    [[nodiscard]] virtual int getLatencySamples() const noexcept { return 0; }

    //==========================================================================
    // Estado / spec (útil para debugging y para bloques que se adaptan por canal)
    [[nodiscard]] bool isPrepared() const noexcept    { return isPreparedFlag; }
    [[nodiscard]] double getSampleRate() const noexcept { return sampleRate; }
    [[nodiscard]] int getMaximumBlockSize() const noexcept { return maximumBlockSize; }
    [[nodiscard]] int getNumChannels() const noexcept { return numChannels; }

protected:
    //==========================================================================
    /*
        prepareInternal()
        Hook para que las clases derivadas:
        - reserven buffers (si aplica),
        - inicialicen filtros,
        - configuren oversampling,
        - precalculen coeficientes,
        etc.

        Reglas:
        - Aquí SÍ puedes asignar memoria y configurar estructuras.
        - En process() NO.
    */
    virtual void prepareInternal (const juce::dsp::ProcessSpec& spec) = 0;

    // Datos disponibles para hijas
    double sampleRate { 44100.0 };
    int maximumBlockSize { 0 };
    int numChannels { 0 };

    // Helpers opcionales para derivadas
    [[nodiscard]] static juce::dsp::ProcessSpec makeSafeSpec (juce::dsp::ProcessSpec spec) noexcept
    {
        // Algunos hosts, al escanear, pueden pasar 0 o valores inválidos.
        if (spec.sampleRate <= 0.0)
            spec.sampleRate = 44100.0;

        if (spec.maximumBlockSize == 0)
            spec.maximumBlockSize = 512;

        if (spec.numChannels == 0)
            spec.numChannels = 2;

        // jassert no cuesta en Release; en Debug te avisa de estados raros.
        jassert (spec.sampleRate > 0.0);
        jassert (spec.maximumBlockSize > 0);
        jassert (spec.numChannels > 0);

        return spec;
    }

private:
    bool bypassedFlag { false };
    bool isPreparedFlag { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DSPBlockBase)
};

} // namespace xb
