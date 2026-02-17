/*
  ==============================================================================

    PluginProcessor.cpp
    Xibalba_Mastering_Suite

    Updated: 2026-02-17
    Author: Mane / Xibalba Studios

    Notes:
    - Conecta LowSculptorEngine en prepareToPlay() y processBlock().
    - TEST: suma completa sin compresión (one-knob en 0).

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

Xibalba_Mastering_SuiteAudioProcessor::Xibalba_Mastering_SuiteAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor (BusesProperties()
        .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
#endif
{
}

Xibalba_Mastering_SuiteAudioProcessor::~Xibalba_Mastering_SuiteAudioProcessor() = default;

const juce::String Xibalba_Mastering_SuiteAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool   Xibalba_Mastering_SuiteAudioProcessor::acceptsMidi() const  { return false; }
bool   Xibalba_Mastering_SuiteAudioProcessor::producesMidi() const { return false; }
bool   Xibalba_Mastering_SuiteAudioProcessor::isMidiEffect() const { return false; }
double Xibalba_Mastering_SuiteAudioProcessor::getTailLengthSeconds() const { return 0.0; }

int  Xibalba_Mastering_SuiteAudioProcessor::getNumPrograms() { return 1; }
int  Xibalba_Mastering_SuiteAudioProcessor::getCurrentProgram() { return 0; }
void Xibalba_Mastering_SuiteAudioProcessor::setCurrentProgram (int) {}
const juce::String Xibalba_Mastering_SuiteAudioProcessor::getProgramName (int) { return {}; }
void Xibalba_Mastering_SuiteAudioProcessor::changeProgramName (int, const juce::String&) {}

void Xibalba_Mastering_SuiteAudioProcessor::prepareToPlay (double sampleRate,
                                                           int samplesPerBlock)
{
    juce::dsp::ProcessSpec spec;
    spec.sampleRate       = sampleRate;
    spec.maximumBlockSize = (juce::uint32) samplesPerBlock;
    spec.numChannels      = (juce::uint32) getTotalNumOutputChannels();

    lowSculptor.prepare (spec);
    lowSculptor.reset();

    // =============================================================
    // 🔬 TEST: SUMA COMPLETA SIN COMPRESIÓN
    // =============================================================
    // En el diseño nuevo: "SQUEEZE" es un knob único musical.
    // 0.0f = bypass auditivo/GR ~ 0 (dependiendo de la implementación interna)
    lowSculptor.setBodySqueezeOneKnob (0.0f);

    lowSculptor.setBodyEnabled (true);
    lowSculptor.setNeckEnabled (true);
}

void Xibalba_Mastering_SuiteAudioProcessor::releaseResources() {}

#ifndef JucePlugin_PreferredChannelConfigurations
bool Xibalba_Mastering_SuiteAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    return layouts.getMainInputChannelSet() == layouts.getMainOutputChannelSet();
}
#endif

void Xibalba_Mastering_SuiteAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                                          juce::MidiBuffer&)
{
    lowSculptor.process (buffer);
}

bool Xibalba_Mastering_SuiteAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* Xibalba_Mastering_SuiteAudioProcessor::createEditor()
{
    return new Xibalba_Mastering_SuiteAudioProcessorEditor (*this);
}

void Xibalba_Mastering_SuiteAudioProcessor::getStateInformation (juce::MemoryBlock&) {}

void Xibalba_Mastering_SuiteAudioProcessor::setStateInformation (const void*, int) {}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Xibalba_Mastering_SuiteAudioProcessor();
}

