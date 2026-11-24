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
    
    RMSPropOptimizer<14> optimizer(0.0005f, 0.99f);
    // GradientDescentOptimizer<14> optimizer(0.0001f);

    NeuralNetworkPopulationHandler<200> populationHandler(14, 1.5f, 0.1f, 1.0f, 1.0f, 1.0f, 1.0f);
    
    const auto tickStart = std::chrono::high_resolution_clock::now();
    
    for (int generation = 0; generation < 1000; ++generation) {
        populationHandler.InitializePopulation();
        populationHandler.ExecutePopulation(optimizer);
    }

    const auto tickEnd = std::chrono::high_resolution_clock::now();
    const auto tickDuration = std::chrono::duration_cast<std::chrono::microseconds>(tickEnd - tickStart);
    std::cout << "Time: " << tickDuration.count() << " microseconds" << std::endl;

    return 0;
}