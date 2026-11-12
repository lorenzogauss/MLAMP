#pragma once

#include <JuceHeader.h>

// Dipendenze locali: percorsi relativi all'header, indipendenti dalle
// "Header Search Paths" dell'IDE.
#include "../libs/RTNeural/modules/json/json.hpp"
#include "../libs/RTNeural/RTNeural/RTNeural.h"

class AudioPluginAudioProcessor : public juce::AudioProcessor
{
public:
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Metodo per caricare il file JSON dei pesi esportato da PyTorch
    void loadModelFromJSON (const juce::File& jsonFile);

private:
    // Definizione dell'architettura speculare al modello Python:
    // LSTM Layer 1: In 1 -> Out 64
    // LSTM Layer 2: In 64 -> Out 64
    // Linear Layer: In 64 -> Out 1
    using ModelType = RTNeural::ModelT<float, 1, 1,
        RTNeural::LSTMLayerT<float, 1, 64>,
        RTNeural::LSTMLayerT<float, 64, 64>,
        RTNeural::DenseT<float, 64, 1>
    >;

    ModelType lstmModel;
    bool isModelLoaded = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
