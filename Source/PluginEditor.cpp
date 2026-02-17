#include "PluginEditor.h"

//==============================================================================
Xibalba_Mastering_SuiteAudioProcessorEditor::Xibalba_Mastering_SuiteAudioProcessorEditor
    (Xibalba_Mastering_SuiteAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setSize (480, 240);
}

Xibalba_Mastering_SuiteAudioProcessorEditor::~Xibalba_Mastering_SuiteAudioProcessorEditor() = default;

void Xibalba_Mastering_SuiteAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::black);

    g.setColour (juce::Colours::white);
    g.setFont (16.0f);
    g.drawFittedText ("Xibalba DSP Sandbox (Mono + Stereo)", getLocalBounds(),
                      juce::Justification::centred, 1);
}

void Xibalba_Mastering_SuiteAudioProcessorEditor::resized()
{
}
