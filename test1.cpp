#include <iostream>
#include <memory>
#include <array>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>
#include <tuple>

#include "Time.hpp"
#include "Neuron.hpp"
#include "NeuralNetwork.hpp"
#include "Data.hpp"
#include "NeuralNetworkTestExecutor.hpp"
#include "NeuralNetworkPopulationHandler.hpp"
#include "GradientDescentOptimizer.hpp"
#include "RMSPropOptimizer.hpp"

int main() {
    std::cout << "Starting up..." << std::endl;
    
    RMSPropOptimizer<14> optimizer(0.005f, 0.95f);
    // GradientDescentOptimizer<14> optimizer(0.00005f);

    NeuralNetworkArchitectureDefinition arcDef 
    {
        .numberOfInputs = 2, 
        .numberOfOutputs = 2,
        .numberInReservoir = 10,
        .inputNeuronConnections = {{3, 7}, {7, 5}},
        .reservoirNeuronConnections = {{3, 7, 5}, {4, 5, 9, 0}, {6, 5, 3, 7}, {5, 7, 6}, {1, 7, 2, 9}, {0, 2, 7}, {1}, {2, 5}, {4, 7}, {8, 2, 5}},
        .outputNeuronConnections = {{3}, {8}}
    };

    NeuralNetworkPopulationHandler<200> populationHandler(arcDef, 14, 1.5f, 0.1f, 1.0f, 1.0f, 1.0f, 1.0f);
    
    const auto tickStart = std::chrono::high_resolution_clock::now();
    
    for (int generation = 0; generation < 80; ++generation) {
        populationHandler.InitializePopulation();
        populationHandler.ExecutePopulation(optimizer);
    }

    const auto tickEnd = std::chrono::high_resolution_clock::now();
    const auto tickDuration = std::chrono::duration_cast<std::chrono::microseconds>(tickEnd - tickStart);
    std::cout << "Time: " << tickDuration.count() << " microseconds" << std::endl;

    populationHandler.PrintAverageNetwork();

    /* Test run found network */
    std::cout << "------------ Test run of mean network ------------\n";
    auto [membraneThresholds, decayRates, spikeStrengths] = populationHandler.GetAverageParameters();

    NeuralNetworkTestExecutor testExecutor(arcDef, membraneThresholds, decayRates, spikeStrengths);
    testExecutor.SetUpNetwork(true);
    testExecutor.Execute(true);

    return 0;
}