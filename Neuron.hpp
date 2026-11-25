#ifndef NEURON_HPP
#define NEURON_HPP

#include <functional>
#include <utility>
#include <vector>

class Neuron
{
    public:
        Neuron() : Neuron(1.0f, 0.1f, 1.0f, []{}) {}
        Neuron(float membraneThreshold, float decayRate, float spikeStrength,
               std::function<void()> spikingCallBack)
            : membraneThreshold{membraneThreshold},
              decayRate{decayRate},
              spikeStrength{spikeStrength},
              spikingCallBack{std::move(spikingCallBack)} {}

        virtual ~Neuron() = default;

        void AddSynapticInput(Neuron* source) {
            sourceNeurons.push_back(source);
            source->AddSynapticOutput(this);
        }

        void AddSynapticOutput(Neuron* destination) {
            destinationNeurons.push_back(destination);
        }

        bool HasSpiked() const {
            return hasSpiked;
        }

        std::vector<Neuron*> GetSourceNeurons() const {
            return sourceNeurons;
        }

        std::vector<Neuron*> GetDestinationNeurons() const {
            return destinationNeurons;
        }

        virtual void ReceiveSpike(float spikeStrength) {
            membranePotential += spikeStrength;
        }

        virtual void Tick(float deltaTime) {
            membranePotential -= membranePotential * decayRate * deltaTime;
            if (membranePotential < 0.0f) {
                membranePotential = 0.0f;
            }

            if (membranePotential >= membraneThreshold) {
                hasSpiked = true;
                for (Neuron* destination : destinationNeurons) {
                    destination->ReceiveSpike(spikeStrength);
                }
                if (spikingCallBack) {
                    spikingCallBack();
                }
                membranePotential = 0.0f;
            }
            else {
                hasSpiked = false;
            }
        }

        float membraneThreshold;
        float decayRate;
        float spikeStrength = 1.0f;
        std::function<void()> spikingCallBack;

    protected:
        bool hasSpiked = false;
        float membranePotential = 0.0f;
        std::vector<Neuron*> sourceNeurons;
        std::vector<Neuron*> destinationNeurons;
};

#endif
