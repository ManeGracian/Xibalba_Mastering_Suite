#pragma once

#include <JuceHeader.h>
#include <atomic>

#include "Plugins/LowSculptor/XC_LowSculptorEngine.h"

class Xibalba_Mastering_SuiteAudioProcessor final
    : public juce::AudioProcessor
{
public:
    Xibalba_Mastering_SuiteAudioProcessor();
    ~Xibalba_Mastering_SuiteAudioProcessor() override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;

    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index,
                            const juce::String& newName) override;

    void prepareToPlay (double sampleRate,
                        int samplesPerBlock) override;

    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout&) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&,
                       juce::MidiBuffer&) override;

    bool hasEditor() const override;
    juce::AudioProcessorEditor* createEditor() override;

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    float getInRmsDb()  const noexcept;
    float getOutRmsDb() const noexcept;

private:
    xb::LowSculptorEngine lowSculptor;

    std::atomic<float> inRmsDb  { -120.0f };
    std::atomic<float> outRmsDb { -120.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR
        (Xibalba_Mastering_SuiteAudioProcessor)
};

