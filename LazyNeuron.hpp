#ifndef LAZY_NEURON_HPP
#define LAZY_NEURON_HPP

#include <cmath>

#include "Neuron.hpp"

class LazyNeuron : public Neuron
{
    public: 
        LazyNeuron() : LazyNeuron(1.0f, 0.1f, 1.0f, []{}) {}
        LazyNeuron(float membraneThreshold, float decayRate, float spikeStrength,
        std::function<void()> spikingCallBack)
            : Neuron(membraneThreshold, decayRate, spikeStrength, std::move(spikingCallBack)) {}

        void ReceiveSpike(float spikeStrength) override {
            currentSpikeStrength += spikeStrength;
            hasSpiked = true;
        }

        void Tick(float deltaTime) override
        {
            if(hasSpiked)
            {
                auto currentPotential = membranePotential*std::expf(-decayRate * lastSpikeTime) + currentSpikeStrength;
                // std::cout << "Current Potential: " << currentPotential << " Membrane Potential: " << membranePotential << std::endl;
                if(currentPotential > membraneThreshold)
                {
                    for (Neuron* destination : destinationNeurons) {
                        destination->ReceiveSpike(spikeStrength);
                    }
                    if (spikingCallBack) {
                        spikingCallBack();
                    }
                    membranePotential = 0.0f;
                }
                else
                {
                    membranePotential = currentPotential;
                }

                currentSpikeStrength = 0.0f;
                lastSpikeTime = 0.0f;
                hasSpiked = false;

                return;
            }
            lastSpikeTime += deltaTime;
        }
    private:
        float currentSpikeStrength = 0.0f;
        float lastSpikeTime = 0.0f;
};

#endif