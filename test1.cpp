#include <iostream>
#include <memory>
#include <array>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <functional>

#include <chrono>
class Time
{
    using nanoseconds = std::chrono::nanoseconds;
    public:
        Time();
        ~Time();
        void Tick()
        {
            lastTime = currentTime;
            currentTime = std::chrono::duration_cast<nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        }

        nanoseconds GetLatestDeltaTime() const
        {
            return currentTime - lastTime;
        }

    private:
        nanoseconds currentTime;
        nanoseconds lastTime;
};

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

class NeuralNetwork
{
    public:
        NeuralNetwork() : NeuralNetwork(0, 0, 0, {}, {}, {}) {} // Default constructor
        NeuralNetwork(int numberOfInputs, int numberOfOutputs, int numberInReservoir,
        const std::vector<float>& membraneThresholds, const std::vector<float>& decayRates, const std::vector<float>& spikeStrengths
        ) :
        numberOfInputs{numberOfInputs}, 
        numberOfOutputs{numberOfOutputs}, 
        numberInReservoir{numberInReservoir},
        inputIndexStart{0},
        outputIndexStart{numberOfInputs},
        reservoirIndexStart{numberOfInputs + numberOfOutputs},
        membraneThresholds{membraneThresholds},
        decayRates{decayRates},
        spikeStrengths{spikeStrengths}
        {}

        void SetUpConnections(int max_connections_per_neuron) {
            InitNeurons();
            SetUpInputConnections();
            SetUpReservoirConnections(max_connections_per_neuron);
            SetUpOutputConnections();
        }

        void SetUpConnections(const std::vector<std::vector<int>>& inputConnections, const std::vector<std::vector<int>>& reservoirConnections, const std::vector<std::vector<int>>& outputConnections)
        {
            InitNeurons();

            for (int inputIndex = 0; inputIndex < numberOfInputs; ++inputIndex) {
                for (int destination : inputConnections[inputIndex]) {
                    neurons[destination + reservoirIndexStart].AddSynapticInput(&neurons[inputIndex + inputIndexStart]);
                }
            }

            for (int reservoirIndex = 0; reservoirIndex < numberInReservoir; ++reservoirIndex) {
                for (int destination : reservoirConnections[reservoirIndex]) {
                    neurons[destination + reservoirIndexStart].AddSynapticInput(&neurons[reservoirIndex + reservoirIndexStart]);
                }
            }

            for (int outputIndex = 0; outputIndex < numberOfOutputs; ++outputIndex) {
                for (int source : outputConnections[outputIndex]) {
                    neurons[outputIndex + outputIndexStart].AddSynapticInput(&neurons[source + reservoirIndexStart]);
                }
            }
        }

        void Tick(float deltaTime) {
            for (int i = 0; i < neuronCount; ++i) {
                neurons[i].Tick(deltaTime);
            }
        }

        Neuron* GetInputNeuron(int index) {
            if (index < 0 || index >= numberOfInputs) {
                throw std::out_of_range("Input neuron index out of range.");
            }
            return &neurons[inputIndexStart + index];
        }

        Neuron* GetOutputNeuron(int index) {
            if (index < 0 || index >= numberOfOutputs) {
                throw std::out_of_range("Output neuron index out of range.");
            }
            return &neurons[outputIndexStart + index];
        }

        void PrintNetworkStructure() const {
            std::cout << "Neural Network Structure:" << std::endl;
            std::cout << "Number of Inputs: " << numberOfInputs << std::endl;
            std::cout << "Number of Outputs: " << numberOfOutputs << std::endl;
            std::cout << "Number in Reservoir: " << numberInReservoir << std::endl;
            std::cout << "Total Neurons: " << neuronCount << std::endl;
            std::cout << "Connections:" << std::endl;
            std::cout << "Input Neurons (" << inputIndexStart << " to " << inputIndexStart + numberOfInputs - 1 << ")" << std::endl;
            for (int i = inputIndexStart; i < inputIndexStart + numberOfInputs; ++i) {
                std::cout << "  Neuron " << i - inputIndexStart << " connects to: ";
                for (const auto& dest : neurons[i].GetDestinationNeurons()) {
                    std::cout << dest - &neurons[0] - reservoirIndexStart << " "; // Print index of destination neuron
                }
                std::cout << std::endl;
            }
            std::cout << "Reservoir Neurons (" << reservoirIndexStart << " to " << reservoirIndexStart + numberInReservoir - 1 << ")" << std::endl;
            for (int i = reservoirIndexStart; i < reservoirIndexStart + numberInReservoir; ++i) {
                std::cout << "  Neuron " << i - reservoirIndexStart << " connects to: ";
                for (const auto& dest : neurons[i].GetDestinationNeurons()) {
                    auto destinationIndex = dest - &neurons[0];
                    if(destinationIndex >= reservoirIndexStart && destinationIndex < reservoirIndexStart + numberInReservoir) {
                        std::cout << destinationIndex - reservoirIndexStart << " "; // Print index of destination neuron
                    }
                    else {
                        std::cout << "Output neuron " << destinationIndex - outputIndexStart << " "; // Print index of destination neuron
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "Output Neurons (" << outputIndexStart << " to " << outputIndexStart + numberOfOutputs - 1 << ")" << std::endl;
            for (int i = outputIndexStart; i < outputIndexStart + numberOfOutputs; ++i) {
                std::cout << "  Neuron " << i - outputIndexStart << " receives input from: ";
                for (const auto& dest : neurons[i].GetSourceNeurons()) {
                    auto sourceIndex = dest - &neurons[0];
                    if(sourceIndex >= reservoirIndexStart && sourceIndex < reservoirIndexStart + numberInReservoir) {
                        std::cout << sourceIndex - reservoirIndexStart << " "; // Print index of source neuron
                    }
                    else {
                        std::cout << "Input neuron " << sourceIndex - inputIndexStart << " "; // Print index of source neuron
                    }
                }
                std::cout << std::endl;
            }
        }

    private:
        Neuron neurons[100]; // Array to hold references to neurons
        int neuronCount = 0; // Count of neurons
        int numberOfInputs;
        int numberOfOutputs;
        int numberInReservoir;
        int inputIndexStart = 0;
        int outputIndexStart = 0;
        int reservoirIndexStart = 0;

        std::vector<float> membraneThresholds;
        std::vector<float> decayRates;  
        std::vector<float> spikeStrengths;

        void InitNeurons()
        {
            // std::cout << "|Input|Output|Reservoir|\n";
            for (int i = 0; i < numberOfInputs + numberOfOutputs + numberInReservoir; ++i)
            {
                if (i > 99)
                {
                    std::cerr << "Neuron count exceeds maximum limit of 100." << std::endl;
                    exit(EXIT_FAILURE);
                }

                const int neuronIndex = neuronCount >= reservoirIndexStart
                                            ? neuronCount - reservoirIndexStart
                                        : neuronCount >= outputIndexStart
                                            ? neuronCount - outputIndexStart
                                            : neuronCount - inputIndexStart;

                const char *neuronType = neuronCount >= reservoirIndexStart
                                             ? "Reservoir"
                                         : neuronCount >= outputIndexStart
                                             ? "Output"
                                             : "Input";

                bool isOutputNeuron = neuronCount >= outputIndexStart && neuronCount < reservoirIndexStart;
                
                const int spaces = neuronCount;

                std::function<void()> spikingCallback = /*[spaces, neuronIndex, neuronType]()
                {
                    std::cout << std::string(spaces, ' ') << neuronIndex << "\t\t\t" << neuronType << " neuron spiked!\n";
                    // std::cout << "Neuron " << neuronIndex << " (" << neuronType << ") spiked!" << std::endl;
                };*/ []{};

                neurons[neuronCount] = Neuron(membraneThresholds[neuronCount], decayRates[neuronCount], spikeStrengths[neuronCount], spikingCallback);
                ++neuronCount;
            }
        }

        void SetUpInputConnections()
        {
            if (numberInReservoir == 0) {
                return;
            }

            std::random_device randomDevice;
            std::mt19937 randomEngine(randomDevice());
            std::uniform_int_distribution<int> reservoirDistribution(
                numberOfInputs + numberOfOutputs,
                numberOfInputs + numberOfOutputs + numberInReservoir - 1);

            for (int inputIndex = 0; inputIndex < numberOfInputs; ++inputIndex) {
                const int reservoirIndex = reservoirDistribution(randomEngine);
                neurons[reservoirIndex].AddSynapticInput(&neurons[inputIndex + inputIndexStart]);
            }
        }

        void SetUpReservoirConnections(int max_connections_per_neuron)
        {
            if (numberInReservoir == 0) {
                return;
            }
            if (max_connections_per_neuron < 1) {
                throw std::invalid_argument("max_connections_per_neuron must be at least 1.");
            }

            std::random_device randomDevice;
            std::mt19937 randomEngine(randomDevice());
            const int reservoirStart = numberOfInputs + numberOfOutputs;
            for (int source = 0; source < numberInReservoir; ++source) {
                std::vector<int> possibleDestinations;
                possibleDestinations.reserve(numberInReservoir);
                for (int destination = 0; destination < numberInReservoir; ++destination) {
                    if (destination != source) {
                        possibleDestinations.push_back(destination);
                    }
                }
                if (possibleDestinations.empty()) {
                    possibleDestinations.push_back(source);
                }

                std::shuffle(possibleDestinations.begin(), possibleDestinations.end(), randomEngine);
                const int maximumConnections = std::min(
                    max_connections_per_neuron,
                    static_cast<int>(possibleDestinations.size()));
                std::uniform_int_distribution<int> connectionCountDistribution(1, maximumConnections);
                const int connectionCount = connectionCountDistribution(randomEngine);
                const int sourceIndex = reservoirStart + source;
                for (int i = 0; i < connectionCount; ++i) {
                    const int destinationIndex = reservoirStart + possibleDestinations[i];
                    neurons[destinationIndex].AddSynapticInput(&neurons[sourceIndex]);
                }
            }
        }

        void SetUpOutputConnections()
        {
            if (numberInReservoir == 0) {
                return;
            }

            std::random_device randomDevice;
            std::mt19937 randomEngine(randomDevice());
            std::uniform_int_distribution<int> reservoirDistribution(
                numberOfInputs + numberOfOutputs,
                numberOfInputs + numberOfOutputs + numberInReservoir - 1);

            for (int outputIndex = 0; outputIndex < numberOfOutputs; ++outputIndex) {
                const int reservoirIndex = reservoirDistribution(randomEngine);
                neurons[outputIndex + outputIndexStart].AddSynapticInput(&neurons[reservoirIndex]);
            }
        }
};

struct Data
{
    // Each batch contains time steps, with one value per input neuron.
    std::vector<std::vector<std::tuple<std::vector<float>, std::vector<float>>>> data = [] {
    std::vector<std::vector<std::vector<float>>> inputData = {
        /* Example 1 in batch*/
        {
            { 0.1f, 0.1f },
            { 0.2f, 0.1f },
            { 0.3f, 0.1f },
            { 0.4f, 0.1f },
            { 0.5f, 0.1f },
            { 0.6f, 0.1f },
            { 0.7f, 0.1f },
            { 0.8f, 0.1f },
            { 0.9f, 0.1f },
            { 0.8f, 0.2f },
            { 0.7f, 0.3f },
            { 0.6f, 0.4f },
            { 0.5f, 0.5f },
            { 0.4f, 0.6f },
            { 0.3f, 0.7f },
            { 0.2f, 0.8f },
            { 0.1f, 0.9f },
            { 0.6f, 0.3f },
            { 0.7f, 0.3f },
            { 0.8f, 0.3f },
            { 0.9f, 0.3f }
        },
        /* Example 2 in batch*/
        {
            { 0.5f, 0.1f },
            { 0.5f, 0.2f },
            { 0.5f, 0.3f },
            { 0.5f, 0.4f },
            { 0.5f, 0.5f },
            { 0.5f, 0.6f },
            { 0.5f, 0.7f },
            { 0.5f, 0.8f },
            { 0.5f, 0.9f }
        }
    };

    // Each batch contains time steps, with one value per output neuron.
    std::vector<std::vector<std::vector<float>>> outputData = {
        /* Example 1 in batch*/
        {
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f },
            { 1.0f, 0.0f }
        },
        /* Example 2 in batch*/
        {
            { 0.0f, 1.0f },
            { 0.0f, 1.0f },
            { 0.0f, 1.0f },
            { 0.0f, 1.0f },
            { 0.0f, 1.0f },
            { 0.0f, 1.0f },
            { 0.0f, 1.0f },
            { 0.0f, 1.0f },
            { 0.0f, 1.0f }
        }
    };

    std::vector<std::vector<std::tuple<std::vector<float>, std::vector<float>>>> combinedData;
    combinedData.reserve(inputData.size());
    for (size_t datapoint = 0; datapoint < inputData.size(); ++datapoint)
    {
        std::vector<std::tuple<std::vector<float>, std::vector<float>>> timeSteps;
        timeSteps.reserve(inputData[datapoint].size());
        for (size_t time = 0; time < inputData[datapoint].size(); ++time)
        {
            timeSteps.emplace_back(inputData[datapoint][time], outputData[datapoint][time]);
        }
        combinedData.push_back(std::move(timeSteps));
    }
    return combinedData;
    }();

    std::vector<std::vector<std::tuple<std::vector<float>, std::vector<float>>>> SampleData() const
    {
        std::random_device randomDevice;
        std::mt19937 randomEngine(randomDevice());

        auto sampledData = data;
        for (auto& datapoint : sampledData)
        {
            for (auto& timeStep : datapoint)
            {
                for (auto* neurons : { &std::get<0>(timeStep), &std::get<1>(timeStep) })
                {
                    for (float& value : *neurons)
                    {
                        std::bernoulli_distribution distribution(value);
                        value = distribution(randomEngine) ? 1.0f : 0.0f;
                    }
                }
            }
        }
        return sampledData;
    }
};

class NeuralNetworkTestExecutor
{
    public:
        NeuralNetworkTestExecutor() {}
        NeuralNetworkTestExecutor(const std::vector<float>& membraneThresholds, const std::vector<float>& decayRates, const std::vector<float>& spikeStrengths)
            : membraneThresholds(membraneThresholds), decayRates(decayRates), spikeStrengths(spikeStrengths) {}
        void SetUpNetwork()
        {
            network = NeuralNetwork(2, 2, 10, membraneThresholds, decayRates, spikeStrengths);
            network.SetUpConnections(inputNeuronConnections, reservoirNeuronConnections, outputNeuronConnections);
        }

        float Execute()
        {
            auto inputNeuron1 = network.GetInputNeuron(0);
            auto inputNeuron2 = network.GetInputNeuron(1);

            auto outputNeuron1 = network.GetOutputNeuron(0);
            auto outputNeuron2 = network.GetOutputNeuron(1);

            Data data{};
            auto sampledData = data.SampleData();

            float totalLoss = 0.0f;

            for (const auto& datapoint : sampledData)
            {
                for (const auto& timeStep : datapoint)
                {
                    const auto& inputNeurons = std::get<0>(timeStep);
                    if(inputNeurons[0] > 0.5f)
                    {
                        inputNeuron1->ReceiveSpike(1.0f);
                    }
                    if(inputNeurons[1] > 0.5f)
                    {
                        inputNeuron2->ReceiveSpike(1.0f);
                    }
                    network.Tick(0.01f);

                    auto neuron1Spiked = network.GetOutputNeuron(0)->HasSpiked() ? 1.0f : 0.0f;
                    auto neuron2Spiked = network.GetOutputNeuron(1)->HasSpiked() ? 1.0f : 0.0f;

                    auto loss1Value = crossEntropyLoss(neuron1Spiked, std::get<1>(timeStep)[0]);
                    auto loss2Value = crossEntropyLoss(neuron2Spiked, std::get<1>(timeStep)[1]);

                    totalLoss += loss1Value + loss2Value;
                }
                totalLoss /= static_cast<float>(datapoint.size());
            }

            return totalLoss;
        }

        float GetMembraneThresholdRegularizationLoss() const
        {
            float loss{0.0f};
            for(float threshold : membraneThresholds)
            {
                loss += (threshold - 1)*(threshold - 1);
            }

            return loss/membraneThresholds.size();
        }

    private:
        /* Architecture hyperparameters */
        NeuralNetwork network;

        std::vector<std::vector<int>> inputNeuronConnections = {{3, 7}, {7, 5}};
        std::vector<std::vector<int>> reservoirNeuronConnections = {{3, 7, 5}, {4, 5, 9, 0}, {6, 5, 3, 7}, {5, 7, 6}, {1, 7, 2, 9}, {0, 2, 7}, {1}, {2, 5}, {4, 7}, {8, 2, 5}};
        std::vector<std::vector<int>> outputNeuronConnections = {{3}, {8}};

        /* Neuron parameters */
        std::vector<float> membraneThresholds;
        std::vector<float> decayRates;
        std::vector<float> spikeStrengths;

        std::function<float(float, float)> crossEntropyLoss = [](float predicted, float target) {
                        return - (target * std::log(predicted + 1e-9f) + (1.0f - target) * std::log(1.0f - predicted + 1e-9f));
                    };
};

template <size_t PopulationSize>
class NeuralNetworkPopulationHandler
{
    public:
        NeuralNetworkPopulationHandler(int numberOfNeurons, float membraneThreshold, float decayRate, float spikeStrength, float membraneThresholdSigma, float decayRateSigma, float spikeStrengthSigma)
            : numberOfNeurons{numberOfNeurons}, population(PopulationSize), membraneThresholdMuInnerGradients(PopulationSize),
            decayRateMuInnerGradients(PopulationSize),
            spikeStrengthMuInnerGradients(PopulationSize),
            membraneThresholdSigmaInnerGradients(PopulationSize),
            decayRateSigmaInnerGradients(PopulationSize),
            spikeStrengthSigmaInnerGradients(PopulationSize),
            membraneThresholdMus(numberOfNeurons, membraneThreshold), 
            decayRateMus(numberOfNeurons, decayRate), 
            spikeStrengthMus(numberOfNeurons, spikeStrength),
            membraneThresholdSigmas(numberOfNeurons, membraneThresholdSigma),
            decayRateSigmas(numberOfNeurons, decayRateSigma),
            spikeStrengthSigmas(numberOfNeurons, spikeStrengthSigma) {}
        
        void InitializePopulation()
        {
            std::random_device randomDevice;
            std::mt19937 randomEngine(randomDevice());

            for (size_t i = 0; i < PopulationSize; ++i)
            {
                std::vector<float> membraneThresholdPertubations(membraneThresholdMus.size());
                std::vector<float> decayRatePertubations(decayRateMus.size());
                std::vector<float> spikeStrengthPertubations(spikeStrengthMus.size());

                membraneThresholdMuInnerGradients[i] = std::vector<float>(membraneThresholdMus.size());
                decayRateMuInnerGradients[i] = std::vector<float>(decayRateMus.size());
                spikeStrengthMuInnerGradients[i] = std::vector<float>(spikeStrengthMus.size());

                membraneThresholdSigmaInnerGradients[i] = std::vector<float>(membraneThresholdMus.size());
                decayRateSigmaInnerGradients[i] = std::vector<float>(decayRateMus.size());
                spikeStrengthSigmaInnerGradients[i] = std::vector<float>(spikeStrengthMus.size());

                std::vector<float> populationMembraneThresholds(numberOfNeurons);
                std::vector<float> populationDecayRates(numberOfNeurons);
                std::vector<float> populationSpikeStrength(numberOfNeurons);

                for (size_t j = 0; j < membraneThresholdMus.size(); ++j)
                {
                    std::normal_distribution<float> membraneThresholdDistribution(membraneThresholdMus[j], membraneThresholdSigmas[j]);
                    std::normal_distribution<float> decayRateDistribution(decayRateMus[j], decayRateSigmas[j]);
                    std::normal_distribution<float> spikeStrengthDistribution(spikeStrengthMus[j], spikeStrengthSigmas[j]);

                    membraneThresholdPertubations[j] = membraneThresholdDistribution(randomEngine);
                    decayRatePertubations[j] = decayRateDistribution(randomEngine);
                    spikeStrengthPertubations[j] = spikeStrengthDistribution(randomEngine);

                    populationMembraneThresholds[j] = membraneThresholdMus[j] + membraneThresholdPertubations[j];
                    populationDecayRates[j] = decayRateMus[j] + decayRatePertubations[j];
                    populationSpikeStrength[j] = spikeStrengthMus[j] + spikeStrengthPertubations[j];

                    membraneThresholdMuInnerGradients[i][j] = muInnerGradient(membraneThresholdSigmas[j], membraneThresholdPertubations[j]);
                    decayRateMuInnerGradients[i][j] = muInnerGradient(decayRateSigmas[j], decayRatePertubations[j]);
                    spikeStrengthMuInnerGradients[i][j] = muInnerGradient(spikeStrengthSigmas[j], spikeStrengthPertubations[j]);

                    membraneThresholdSigmaInnerGradients[i][j] = sigmaInnerGradient(membraneThresholdSigmas[j], membraneThresholdPertubations[j]);
                    decayRateSigmaInnerGradients[i][j] = sigmaInnerGradient(decayRateSigmas[j], decayRatePertubations[j]);
                    spikeStrengthSigmaInnerGradients[i][j] = sigmaInnerGradient(spikeStrengthSigmas[j], spikeStrengthPertubations[j]);
                }



                population[i] = NeuralNetworkTestExecutor{populationMembraneThresholds, populationDecayRates, populationSpikeStrength};
                population[i].SetUpNetwork();
            }
        }

        void ExecutePopulation(auto optimizer)
        {
            std::vector<float> membraneMuGradient(membraneThresholdMus.size(), 0.0f);
            std::vector<float> membraneSigmaGradient(membraneThresholdSigmas.size(), 0.0f);
            std::vector<float> decayRateMuGradient(decayRateMus.size(), 0.0f);
            std::vector<float> decayRateSigmaGradient(decayRateSigmas.size(), 0.0f);
            std::vector<float> spikeStrengthMuGradient(spikeStrengthMus.size(), 0.0f);
            std::vector<float> spikeStrengthSigmaGradient(spikeStrengthSigmas.size(), 0.0f);

            float totalPopulationLoss = 0.0f;

            for (size_t i = 0; i < PopulationSize; ++i)
            {
                auto populationLoss = population[i].Execute();
                auto membraneRegularizationLoss = population[i].GetMembraneThresholdRegularizationLoss();
                totalPopulationLoss += populationLoss + 10.0f*membraneRegularizationLoss;

                for(size_t j = 0; j < membraneThresholdMus.size(); ++j)
                {
                    membraneMuGradient[j] += populationLoss*membraneThresholdMuInnerGradients[i][j];
                    membraneSigmaGradient[j] += populationLoss*membraneThresholdSigmaInnerGradients[i][j];
                    decayRateMuGradient[j] += populationLoss*decayRateMuInnerGradients[i][j];
                    decayRateSigmaGradient[j] += populationLoss*decayRateSigmaInnerGradients[i][j];
                    spikeStrengthMuGradient[j] += populationLoss*spikeStrengthMuInnerGradients[i][j];
                    spikeStrengthSigmaGradient[j] += populationLoss*spikeStrengthSigmaInnerGradients[i][j];
                }
            }

            totalPopulationLoss /= static_cast<float>(PopulationSize);
            int discreteLoss = static_cast<int>(totalPopulationLoss); // Convert to int
            std::cout << std::string(discreteLoss, ' ') << "* " << totalPopulationLoss << std::endl;

            /* Normalize gradients with population size */
            for(size_t j = 0; j < membraneThresholdMus.size(); ++j)
            {
                membraneMuGradient[j] /= PopulationSize;
                membraneSigmaGradient[j] /= PopulationSize;
                decayRateMuGradient[j] /= PopulationSize;
                decayRateSigmaGradient[j] /= PopulationSize;
                spikeStrengthMuGradient[j] /= PopulationSize;
                spikeStrengthSigmaGradient[j] /= PopulationSize;
            }

            // const float learningRate = 0.01f; // Define a learning rate for gradient descent
            // float meanMembraneThresholdMu{0.0f};
            // float meanMembraneThresholdSigma{0.0f};
            // float meanDecayRateMu{0.0f};
            // float meanDecayRateSigma{0.0f};
            // float meanSpikeStrengthMu{0.0f};
            // float meanSpikeStrengthSigma{0.0f};

            optimizer.ApplyGradients(
                membraneThresholdMus, membraneThresholdSigmas, membraneMuGradient, membraneSigmaGradient,
                decayRateMus, decayRateSigmas, decayRateMuGradient, decayRateSigmaGradient,
                spikeStrengthMus, spikeStrengthSigmas, spikeStrengthMuGradient, spikeStrengthSigmaGradient);

            /* Apply gradients v0.1 */
            

            // meanMembraneThresholdMu /= membraneThresholdMus.size();
            // meanMembraneThresholdSigma /= membraneThresholdSigmas.size();
            // meanDecayRateMu /= decayRateMus.size();
            // meanDecayRateSigma /= decayRateSigmas.size();
            // meanSpikeStrengthMu /= spikeStrengthMus.size();
            // meanSpikeStrengthSigma /= spikeStrengthSigmas.size();
            // std::cout << "Mean Membrane Threshold Mu: " << meanMembraneThresholdMu << std::endl;
        }

    private:
        int numberOfNeurons;
        std::vector<NeuralNetworkTestExecutor> population;

        std::vector<std::vector<float>> membraneThresholdMuInnerGradients;
        std::vector<std::vector<float>> decayRateMuInnerGradients;
        std::vector<std::vector<float>> spikeStrengthMuInnerGradients;

        std::vector<std::vector<float>> membraneThresholdSigmaInnerGradients;
        std::vector<std::vector<float>> decayRateSigmaInnerGradients;
        std::vector<std::vector<float>> spikeStrengthSigmaInnerGradients;

        std::vector<float> membraneThresholdMus;
        std::vector<float> decayRateMus;
        std::vector<float> spikeStrengthMus;

        std::vector<float> membraneThresholdSigmas;
        std::vector<float> decayRateSigmas;
        std::vector<float> spikeStrengthSigmas;

        std::function<float(float, float)> muInnerGradient = [](float sigma, float epsilon) {
            return epsilon/sigma/sigma;
        };
        std::function<float(float, float)> sigmaInnerGradient = [](float sigma, float epsilon) {
            return (epsilon*epsilon/sigma/sigma - 1);
        };
};

template<size_t numberOfNeurons>
class GradientDescentOptimizer
{
    public:
        GradientDescentOptimizer() : GradientDescentOptimizer(0.01f) {}
        GradientDescentOptimizer(float learningRate = 0.01f) :
        learningRate{learningRate} {}

        void ApplyGradients(
                std::vector<float>& membraneThresholdMus, 
                std::vector<float>& membraneThresholdSigmas, 
                const std::vector<float>& membraneMuGradient, 
                const std::vector<float>& membraneSigmaGradient,
                std::vector<float>& decatRateMus, 
                std::vector<float>& decayRateSigmas, 
                const std::vector<float>& decayRateMuGradient, 
                const std::vector<float>& decayRateSigmaGradient,
                std::vector<float>& spikeStrengthMus, 
                std::vector<float>& spikeStrengthSigmas, 
                const std::vector<float>& spikeStrengthMuGradient, 
                const std::vector<float>& spikeStrengthSigmaGradient)
        {
            for(size_t j = 0; j < numberOfNeurons; ++j)
            {
                membraneThresholdMus[j] -= getGradientStep(membraneMuGradient[j]);
                membraneThresholdSigmas[j] -= getGradientStep(membraneSigmaGradient[j]);
                decatRateMus[j] -= getGradientStep(decayRateMuGradient[j]);
                decayRateSigmas[j] -= getGradientStep(decayRateSigmaGradient[j]);
                spikeStrengthMus[j] -= getGradientStep(spikeStrengthMuGradient[j]);
                spikeStrengthSigmas[j] -= getGradientStep(spikeStrengthSigmaGradient[j]);
            }

            for(size_t j = 0; j < numberOfNeurons; ++j)
            {
                /* Ensure sigmas remain positive */
                if(membraneThresholdSigmas[j] < 0.0001f) membraneThresholdSigmas[j] = 0.0001f;
                if(decayRateSigmas[j] < 0.0001f) decayRateSigmas[j] = 0.0001f;
                if(spikeStrengthSigmas[j] < 0.0001f) spikeStrengthSigmas[j] = 0.0001f;
            }
        }

    private:
        float learningRate;
        std::function<float(float)> getGradientStep = [this](float grad) { return this->learningRate*grad; };
};

template<size_t numberOfNeurons>
class RMSPropOptimizer
{
    public:
        RMSPropOptimizer() : RMSPropOptimizer(0.01f, 0.9f) {}
        RMSPropOptimizer(float learningRate = 0.01f, float decay = 0.9f) : 
            learningRate{learningRate},
            decay{decay},
            membraneThresholdMuSpeedAverage(numberOfNeurons, 0.0f),
            membraneThresholdSigmasSpeedAverage(numberOfNeurons, 0.0f),
            decayRateMusSpeedAverage(numberOfNeurons, 0.0f),
            decayRateSigmasSpeedAverage(numberOfNeurons, 0.0f),
            spikeStrengthMusSpeedAverage(numberOfNeurons, 0.0f),
            spikeStrengthSigmasSpeedAverage(numberOfNeurons, 0.0f)
            {}
        void ApplyGradients(
                std::vector<float>& membraneThresholdMus, 
                std::vector<float>& membraneThresholdSigmas, 
                const std::vector<float>& membraneMuGradient, 
                const std::vector<float>& membraneSigmaGradient,
                std::vector<float>& decatRateMus, 
                std::vector<float>& decayRateSigmas, 
                const std::vector<float>& decayRateMuGradient, 
                const std::vector<float>& decayRateSigmaGradient,
                std::vector<float>& spikeStrengthMus, 
                std::vector<float>& spikeStrengthSigmas, 
                const std::vector<float>& spikeStrengthMuGradient, 
                const std::vector<float>& spikeStrengthSigmaGradient)
        {
            for(size_t j = 0; j < numberOfNeurons; ++j)
            {
                membraneThresholdMuSpeedAverage[j] = updateMovingAverage(membraneMuGradient[j], membraneThresholdMuSpeedAverage[j]);
                membraneThresholdSigmasSpeedAverage[j] = updateMovingAverage(membraneSigmaGradient[j], membraneThresholdSigmasSpeedAverage[j]);
                decayRateMusSpeedAverage[j] = updateMovingAverage(decayRateMuGradient[j], decayRateMusSpeedAverage[j]);
                decayRateSigmasSpeedAverage[j] = updateMovingAverage(decayRateSigmaGradient[j], decayRateSigmasSpeedAverage[j]);
                spikeStrengthMusSpeedAverage[j] = updateMovingAverage(spikeStrengthMuGradient[j], spikeStrengthMusSpeedAverage[j]);
                spikeStrengthSigmasSpeedAverage[j] = updateMovingAverage(spikeStrengthSigmaGradient[j], spikeStrengthSigmasSpeedAverage[j]);
            }

            for(size_t j = 0; j < numberOfNeurons; ++j)
            {
                membraneThresholdMus[j] -= getGradientStep(membraneMuGradient[j], membraneThresholdMuSpeedAverage[j]);
                membraneThresholdSigmas[j] -= getGradientStep(membraneSigmaGradient[j], membraneThresholdSigmasSpeedAverage[j]);
                decatRateMus[j] -= getGradientStep(decayRateMuGradient[j], decayRateMusSpeedAverage[j]);
                decayRateSigmas[j] -= getGradientStep(decayRateSigmaGradient[j], decayRateSigmasSpeedAverage[j]);
                spikeStrengthMus[j] -= getGradientStep(spikeStrengthMuGradient[j], spikeStrengthMusSpeedAverage[j]);
                spikeStrengthSigmas[j] -= getGradientStep(spikeStrengthSigmaGradient[j], spikeStrengthSigmasSpeedAverage[j]);
            }

            for(size_t j = 0; j < numberOfNeurons; ++j)
            {
                /* Ensure sigmas remain positive */
                if(membraneThresholdSigmas[j] < 0.00001f) membraneThresholdSigmas[j] = 0.00001f;
                if(decayRateSigmas[j] < 0.00001f) decayRateSigmas[j] = 0.00001f;
                if(spikeStrengthSigmas[j] < 0.00001f) spikeStrengthSigmas[j] = 0.00001f;
            }
        }
    
    private:
        float learningRate;
        float decay;
        std::vector<float> membraneThresholdMuSpeedAverage;
        std::vector<float> membraneThresholdSigmasSpeedAverage;
        std::vector<float> decayRateMusSpeedAverage;
        std::vector<float> decayRateSigmasSpeedAverage;
        std::vector<float> spikeStrengthMusSpeedAverage;
        std::vector<float> spikeStrengthSigmasSpeedAverage;
        std::function<float(float, float)> updateMovingAverage = [this](float grad, float oldAvg) { return this->decay*oldAvg + grad*grad*(1.0f-this->decay); };
        std::function<float(float, float)> getGradientStep = [this](float grad, float avg) { return this->learningRate*grad/(sqrtf(avg) + 0.0001f); };
};

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