/*
  ==============================================================================

    PluginProcessor.h
    Xibalba_Mastering_Suite

    Updated: 2026-02-16 20:00 (local)
    Author:  Mane / Xibalba Studios

    Notes:
    - Conectamos el engine "LowSculptor" en modo passthrough (no procesa aún).
    - Esto es el inicio del refactor PRO: PluginProcessor delega DSP al engine.
    - RT-Safe: el engine no hace allocations en process().

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>

// Engine del plugin (por ahora passthrough)
#include "Plugins/LowSculptor/XC_LowSculptorEngine.h"

//==============================================================================
class Xibalba_Mastering_SuiteAudioProcessor final : public juce::AudioProcessor
{
public:
    //==============================================================================
    Xibalba_Mastering_SuiteAudioProcessor();
    ~Xibalba_Mastering_SuiteAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;

    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

private:
    //==============================================================================
    // Engine principal (por ahora passthrough). Más adelante encapsula:
    // IN Trim/Meter → Crossover → BODY chain → NECK chain → SUM → OUT Gain/Meter.
    xb::LowSculptorEngine lowSculptor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Xibalba_Mastering_SuiteAudioProcessor)
};