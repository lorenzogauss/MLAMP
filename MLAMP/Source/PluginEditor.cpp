#include "PluginProcessor.h"
#include "PluginEditor.h"

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Configurazione del pulsante Carica Modello
    loadModelButton.addListener(this);
    addAndMakeVisible(loadModelButton);

    // Configurazione dell'etichetta di stato
    statusLabel.setText("Seleziona un modello JSON per iniziare", juce::dontSendNotification);
    statusLabel.setJustificationType(juce::Justification::centred);
    statusLabel.setFont(juce::FontOptions(14.0f));
    addAndMakeVisible(statusLabel);

    // Dimensione fissa della finestra del Plugin
    setSize (400, 200);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
    loadModelButton.removeListener(this);
}

void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    // Sfondo del Plugin
    g.fillAll (juce::Colour(0xff222222));

    // Titolo
    g.setColour (juce::Colours::white);
    g.setFont (juce::FontOptions(20.0f, juce::Font::bold));
    g.drawText ("MLAMP", getLocalBounds().removeFromTop(50), juce::Justification::centred, true);
}

void AudioPluginAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced(20);
    area.removeFromTop(40); // Spazio per il titolo

    loadModelButton.setBounds(area.removeFromTop(40).reduced(40, 0));
    area.removeFromTop(10);
    statusLabel.setBounds(area.removeFromTop(30));
}

void AudioPluginAudioProcessorEditor::buttonClicked (juce::Button* button)
{
    if (button == &loadModelButton)
    {
        // Apertura asincrona del file per evitare il blocco del thread grafico
        fileChooser = std::make_unique<juce::FileChooser> (
            "Seleziona i pesi model_weights.json",
            juce::File::getSpecialLocation(juce::File::userHomeDirectory),
            "*.json");

        auto folderChooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

        fileChooser->launchAsync (folderChooserFlags, [this] (const juce::FileChooser& chooser)
        {
            auto file = chooser.getResult();
            if (file.existsAsFile())
            {
                // Carica il modello nel Processor
                audioProcessor.loadModelFromJSON(file);
                
                statusLabel.setText("Modello Caricato: " + file.getFileName(), juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
            }
        });
    }
}
