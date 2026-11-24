#include <functional>

class Neuron
{
    public: 
        Neuron() : Neuron(1.0f, 0.1f, 1.0f, []{}) {} // Default constructor with example values
        Neuron(float membraneThreshold, float decayRate, float spikeStrength,
        std::function<void()> spikingCallBack) : 
            membraneThreshold{membraneThreshold},
            decayRate{decayRate},
            spikeStrength{spikeStrength},
            spikingCallBack{std::move(spikingCallBack)} {}

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

        void ReceiveSpike(float spikeStrength) {
            membranePotential += spikeStrength;
        }

        /* Another spiking algorithm in comments */
        // void ReceiveSpike(float spikeStrength) {
        //     currentSpikeStrength += spikeStrength;
        //     hasSpiked = true;
        // }

        void Tick(float deltaTime) {
            membranePotential -= membranePotential * decayRate * deltaTime;
            if (membranePotential < 0.0f) {
                membranePotential = 0.0f;
            }
        
            if (membranePotential >= membraneThreshold) {
                hasSpiked = true;
                for (Neuron* dest : destinationNeurons) {
                    dest->ReceiveSpike(spikeStrength);
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

        // void Tick(float deltaTime)
        // {
        //     if(hasSpiked)
        //     {
        //         auto currentPotential = membranePotential*std::expf(-decayRate * lastSpikeTime) + currentSpikeStrength;
        //         // std::cout << "Current Potential: " << currentPotential << " Membrane Potential: " << membranePotential << std::endl;
        //         if(currentPotential > membraneThreshold)
        //         {
        //             for (Neuron* dest : destinationNeurons) {
        //                 dest->ReceiveSpike(spikeStrength);
        //             }
        //             if (spikingCallBack) {
        //                 spikingCallBack();
        //             }
        //             membranePotential = 0.0f;
        //         }
        //         else
        //         {
        //             membranePotential = currentPotential;
        //         }

        //         currentSpikeStrength = 0.0f;
        //         lastSpikeTime = 0.0f;
        //         hasSpiked = false;

        //         return;
        //     }
        //     lastSpikeTime += deltaTime;
        // }

        float membraneThreshold;
        float decayRate;
        float spikeStrength = 1.0f; // Default spike strength
        std::function<void()> spikingCallBack;
    private:
        bool hasSpiked = false;
        // float currentSpikeStrength = 0.0f;
        float membranePotential = 0.0f;
        // float lastSpikeTime = 0.0f;
        std::vector<Neuron*> sourceNeurons; // Non-owning references to source neurons
        std::vector<Neuron*> destinationNeurons; // Non-owning references to destination neurons
};