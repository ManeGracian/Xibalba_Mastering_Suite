#include "PluginEditor.h"
#include "PluginProcessor.h"

//==============================================================================
Xibalba_Mastering_SuiteAudioProcessorEditor::
Xibalba_Mastering_SuiteAudioProcessorEditor
    (Xibalba_Mastering_SuiteAudioProcessor& p)
    : AudioProcessorEditor (&p),
      audioProcessor (p)
{
    setSize (400, 200);
}

//==============================================================================
Xibalba_Mastering_SuiteAudioProcessorEditor::
~Xibalba_Mastering_SuiteAudioProcessorEditor() = default;

//==============================================================================
void Xibalba_Mastering_SuiteAudioProcessorEditor::
paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::black);

    g.setColour (juce::Colours::white);
    g.setFont (15.0f);

    g.drawFittedText ("LowSculptor Dev - RS124 BODY",
                      getLocalBounds(),
                      juce::Justification::centred,
                      1);
}

//==============================================================================
void Xibalba_Mastering_SuiteAudioProcessorEditor::
resized()
{
}
