/*
  ==============================================================================

    PluginProcessor.cpp
    Xibalba_Mastering_Suite

    Updated: 2026-02-16 20:00 (local)
    Author:  Mane / Xibalba Studios

    Notes:
    - PASO 4: conectamos LowSculptorEngine en prepareToPlay() y processBlock().
    - El engine aún es passthrough; esto solo valida el cableado.
    - Mantiene soporte MONO (1,1) y STEREO (2,2) sin 1→2.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
Xibalba_Mastering_SuiteAudioProcessor::Xibalba_Mastering_SuiteAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor (BusesProperties()
                      // Default STEREO (mejor para Ableton). Aún así soportamos MONO.
                      .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
#endif
{
}

Xibalba_Mastering_SuiteAudioProcessor::~Xibalba_Mastering_SuiteAudioProcessor() = default;

//==============================================================================
const juce::String Xibalba_Mastering_SuiteAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool Xibalba_Mastering_SuiteAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool Xibalba_Mastering_SuiteAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool Xibalba_Mastering_SuiteAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double Xibalba_Mastering_SuiteAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int Xibalba_Mastering_SuiteAudioProcessor::getNumPrograms()
{
    return 1;
}

int Xibalba_Mastering_SuiteAudioProcessor::getCurrentProgram()
{
    return 0;
}

void Xibalba_Mastering_SuiteAudioProcessor::setCurrentProgram (int)
{
}

const juce::String Xibalba_Mastering_SuiteAudioProcessor::getProgramName (int)
{
    return {};
}

void Xibalba_Mastering_SuiteAudioProcessor::changeProgramName (int, const juce::String&)
{
}

//==============================================================================
void Xibalba_Mastering_SuiteAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    //==============================================================
    // Creamos ProcessSpec para el engine (estándar JUCE dsp).
    // NOTA: usamos outputChannels porque es lo que realmente procesamos hacia el host.
    juce::dsp::ProcessSpec spec;
    spec.sampleRate       = sampleRate;
    spec.maximumBlockSize = (juce::uint32) juce::jmax (1, samplesPerBlock);
    spec.numChannels      = (juce::uint32) juce::jmax (1, getTotalNumOutputChannels());

    // Preparar y resetear el engine (todo alloc/init debe ocurrir aquí, no en process()).
    lowSculptor.prepare (spec);
    lowSculptor.reset();
}

void Xibalba_Mastering_SuiteAudioProcessor::releaseResources()
{
    // Si en el futuro tu engine necesita liberar cosas, lo llamas aquí.
    // lowSculptor.releaseResources();  // (no existe aún)
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool Xibalba_Mastering_SuiteAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& inSet  = layouts.getChannelSet (true,  0);
    const auto& outSet = layouts.getChannelSet (false, 0);

    // Solo soportamos:
    //  - MONO->MONO
    //  - STEREO->STEREO
    const bool monoOK =
        (inSet  == juce::AudioChannelSet::mono()
      && outSet == juce::AudioChannelSet::mono());

    const bool stereoOK =
        (inSet  == juce::AudioChannelSet::stereo()
      && outSet == juce::AudioChannelSet::stereo());

    return monoOK || stereoOK;
}
#endif

void Xibalba_Mastering_SuiteAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                                         juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused (midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const int inCh  = getTotalNumInputChannels();
    const int outCh = getTotalNumOutputChannels();
    const int n     = buffer.getNumSamples();

    // En {1,1},{2,2} normalmente inCh == outCh,
    // pero limpiamos por seguridad cualquier canal extra.
    for (int ch = inCh; ch < outCh; ++ch)
        buffer.clear (ch, 0, n);

    //==============================================================
    // PASO 4: delegamos al engine (por ahora passthrough).
    lowSculptor.process (buffer);
}

//==============================================================================
bool Xibalba_Mastering_SuiteAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* Xibalba_Mastering_SuiteAudioProcessor::createEditor()
{
    return new Xibalba_Mastering_SuiteAudioProcessorEditor (*this);
}

//==============================================================================
void Xibalba_Mastering_SuiteAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // Estado dummy por ahora.
    juce::MemoryOutputStream mos (destData, true);
    mos.writeInt (1); // version
}

void Xibalba_Mastering_SuiteAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    juce::MemoryInputStream mis (data, (size_t) sizeInBytes, false);
    juce::ignoreUnused (mis);
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Xibalba_Mastering_SuiteAudioProcessor();
}