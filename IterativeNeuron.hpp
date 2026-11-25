#ifndef ITERATIVE_NEURON_HPP
#define ITERATIVE_NEURON_HPP

#include "Neuron.hpp"

class IterativeNeuron : public Neuron
{
    public: 
        IterativeNeuron() : IterativeNeuron(1.0f, 0.1f, 1.0f, []{}) {}
        IterativeNeuron(float membraneThreshold, float decayRate, float spikeStrength,
        std::function<void()> spikingCallBack)
            : Neuron(membraneThreshold, decayRate, spikeStrength, std::move(spikingCallBack)) {}

        void ReceiveSpike(float spikeStrength) override {
            membranePotential += spikeStrength;
        }

        void Tick(float deltaTime) override {
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
};

#endif