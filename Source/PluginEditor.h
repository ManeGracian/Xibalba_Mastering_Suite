#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class Xibalba_Mastering_SuiteAudioProcessorEditor final
    : public juce::AudioProcessorEditor
{
public:
    explicit Xibalba_Mastering_SuiteAudioProcessorEditor
        (Xibalba_Mastering_SuiteAudioProcessor&);
    ~Xibalba_Mastering_SuiteAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    Xibalba_Mastering_SuiteAudioProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR
        (Xibalba_Mastering_SuiteAudioProcessorEditor)
};
