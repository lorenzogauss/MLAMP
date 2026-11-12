#include "PluginProcessor.h"
#include "PluginEditor.h"

AudioPluginAudioProcessor::AudioPluginAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       )
#endif
{
    // Tenta di caricare il modello direttamente dalla Home Directory dell'utente all'avvio
    auto homeDir = juce::File::getSpecialLocation(juce::File::userHomeDirectory);
    auto weightsFile = homeDir.getChildFile("model_weights.json");

    if (weightsFile.existsAsFile())
    {
        loadModelFromJSON(weightsFile);
    }
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
}

const juce::String AudioPluginAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const { return false; }
bool AudioPluginAudioProcessor::producesMidi() const { return false; }
bool AudioPluginAudioProcessor::isMidiEffect() const { return false; }
double AudioPluginAudioProcessor::getTailLengthSeconds() const { return 0.0; }

int AudioPluginAudioProcessor::getNumPrograms() { return 1; }
int AudioPluginAudioProcessor::getCurrentProgram() { return 0; }
void AudioPluginAudioProcessor::setCurrentProgram (int index) {}
const juce::String AudioPluginAudioProcessor::getProgramName (int index) { return {}; }
void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName) {}

void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Reset dello stato della rete neurale per pulire la memoria interna prima di riprodurre
    lstmModel.reset();
}

void AudioPluginAudioProcessor::releaseResources()
{
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    return true;
}

void AudioPluginAudioProcessor::loadModelFromJSON (const juce::File& jsonFile)
{
    if (!jsonFile.existsAsFile())
        return;

    try
    {
        // Lettura e parsing dei pesi mediante il parser JSON nativo di RTNeural
        auto jsonStream = jsonFile.createInputStream();
        if (jsonStream != nullptr)
        {
            auto jsonString = jsonStream->readEntireStreamAsString().toStdString();
            auto jsonObject = nlohmann::json::parse(jsonString);

            // Mapping automatico dei pesi salvati da PyTorch nei layer di RTNeural
            auto& lstm1 = lstmModel.get<0>();
            auto& lstm2 = lstmModel.get<1>();
            auto& dense = lstmModel.get<2>();

            // Assegnazione dei pesi dai dizionari salvati in Python
            RTNeural::torch_helpers::loadLSTM<float>(jsonObject, "lstm.", lstm1, true, 0);
            RTNeural::torch_helpers::loadLSTM<float>(jsonObject, "lstm.", lstm2, true, 1);
            RTNeural::torch_helpers::loadDense<float>(jsonObject, "fc.", dense);

            lstmModel.reset();
            isModelLoaded = true;
        }
    }
    catch (const std::exception& e)
    {
        juce::Logger::writeToLog("Errore durante il caricamento del modello: " + juce::String(e.what()));
        isModelLoaded = false;
    }
}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    // FONDAMENTALE: Previene il system overload disabilitando i denormali nei momenti di silenzio
    juce::ScopedNoDenormals noDenormals;

    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // Se il modello non è caricato, passa l'audio in bypass
    if (!isModelLoaded)
        return;

    auto* channelDataLeft = buffer.getWritePointer(0);
    const int numSamples = buffer.getNumSamples();

    // Processamento sample-by-sample zero-allocazione nel thread audio
    for (int sample = 0; sample < numSamples; ++sample)
    {
        float inputSample = channelDataLeft[sample];
        float inArray[1] = { inputSample };

        // Inferenza ad alta velocità ottimizzata SIMD
        float outputSample = lstmModel.forward(inArray);

        channelDataLeft[sample] = outputSample;
    }

    // Copia il risultato sul canale destro se la traccia è impostata in Stereo
    if (totalNumOutputChannels > 1)
    {
        auto* channelDataRight = buffer.getWritePointer(1);
        std::copy_n(channelDataLeft, numSamples, channelDataRight);
    }
}

bool AudioPluginAudioProcessor::hasEditor() const { return true; }
juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    return new AudioPluginAudioProcessorEditor (*this);
}

void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData) {}
void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes) {}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}
