#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class AudioPluginAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                         private juce::Button::Listener
{
public:
    AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void buttonClicked (juce::Button* button) override;

private:
    // Riferimento al processor
    AudioPluginAudioProcessor& audioProcessor;

    // Elementi GUI
    juce::TextButton loadModelButton { "Carica Modello JSON" };
    juce::Label statusLabel;

    // File Chooser gestito in modo sicuro (senza bloccare il thread GUI)
    std::unique_ptr<juce::FileChooser> fileChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
