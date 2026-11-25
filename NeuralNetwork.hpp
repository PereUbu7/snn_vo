#include "IterativeNeuron.hpp"

struct NeuralNetworkArchitectureDefinition
{
    int numberOfInputs;
    int numberOfOutputs;
    int numberInReservoir;
    std::vector<std::vector<int>> inputNeuronConnections;
    std::vector<std::vector<int>> reservoirNeuronConnections;
    std::vector<std::vector<int>> outputNeuronConnections;
};

class NeuralNetwork
{
    public:
        NeuralNetwork() : NeuralNetwork({0, 0, 0}, {}, {}, {}) {} // Default constructor
        NeuralNetwork(NeuralNetworkArchitectureDefinition architectureDefinition,
        const std::vector<float>& membraneThresholds, const std::vector<float>& decayRates, const std::vector<float>& spikeStrengths
        ) :
        architectureDefinition{architectureDefinition},
        inputIndexStart{0},
        outputIndexStart{architectureDefinition.numberOfInputs},
        reservoirIndexStart{architectureDefinition.numberOfInputs + architectureDefinition.numberOfOutputs},
        membraneThresholds{membraneThresholds},
        decayRates{decayRates},
        spikeStrengths{spikeStrengths}
        {}

        void SetUpConnections(int max_connections_per_neuron, bool verbose = false) {
            InitNeurons(verbose);
            SetUpInputConnections();
            SetUpReservoirConnections(max_connections_per_neuron);
            SetUpOutputConnections();
        }

        void SetUpConnections(bool verbose = false)
        {
            InitNeurons(verbose);

            for (int inputIndex = 0; inputIndex < architectureDefinition.numberOfInputs; ++inputIndex) {
                for (int destination : architectureDefinition.inputNeuronConnections[inputIndex]) {
                    neurons[destination + reservoirIndexStart].AddSynapticInput(&neurons[inputIndex + inputIndexStart]);
                }
            }

            for (int reservoirIndex = 0; reservoirIndex < architectureDefinition.numberInReservoir; ++reservoirIndex) {
                for (int destination : architectureDefinition.reservoirNeuronConnections[reservoirIndex]) {
                    neurons[destination + reservoirIndexStart].AddSynapticInput(&neurons[reservoirIndex + reservoirIndexStart]);
                }
            }

            for (int outputIndex = 0; outputIndex < architectureDefinition.numberOfOutputs; ++outputIndex) {
                for (int source : architectureDefinition.outputNeuronConnections[outputIndex]) {
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
            if (index < 0 || index >= architectureDefinition.numberOfInputs) {
                throw std::out_of_range("Input neuron index out of range.");
            }
            return &neurons[inputIndexStart + index];
        }

        Neuron* GetOutputNeuron(int index) {
            if (index < 0 || index >= architectureDefinition.numberOfOutputs) {
                throw std::out_of_range("Output neuron index out of range.");
            }
            return &neurons[outputIndexStart + index];
        }

        void PrintNetworkStructure() const {
            std::cout << "Neural Network Structure:" << std::endl;
            std::cout << "Number of Inputs: " << architectureDefinition.numberOfInputs << std::endl;
            std::cout << "Number of Outputs: " << architectureDefinition.numberOfOutputs << std::endl;
            std::cout << "Number in Reservoir: " << architectureDefinition.numberInReservoir << std::endl;
            std::cout << "Total Neurons: " << neuronCount << std::endl;
            std::cout << "Connections:" << std::endl;
            std::cout << "Input Neurons (" << inputIndexStart << " to " << inputIndexStart + architectureDefinition.numberOfInputs - 1 << ")" << std::endl;
            for (int i = inputIndexStart; i < inputIndexStart + architectureDefinition.numberOfInputs; ++i) {
                std::cout << "  Neuron " << i - inputIndexStart << " connects to: ";
                for (const auto& dest : neurons[i].GetDestinationNeurons()) {
                    std::cout << dest - &neurons[0] - reservoirIndexStart << " "; // Print index of destination neuron
                }
                std::cout << std::endl;
            }
            std::cout << "Reservoir Neurons (" << reservoirIndexStart << " to " << reservoirIndexStart + architectureDefinition.numberInReservoir - 1 << ")" << std::endl;
            for (int i = reservoirIndexStart; i < reservoirIndexStart + architectureDefinition.numberInReservoir; ++i) {
                std::cout << "  Neuron " << i - reservoirIndexStart << " connects to: ";
                for (const auto& dest : neurons[i].GetDestinationNeurons()) {
                    auto destinationIndex = dest - &neurons[0];
                    if(destinationIndex >= reservoirIndexStart && destinationIndex < reservoirIndexStart + architectureDefinition.numberInReservoir) {
                        std::cout << destinationIndex - reservoirIndexStart << " "; // Print index of destination neuron
                    }
                    else {
                        std::cout << "Output neuron " << destinationIndex - outputIndexStart << " "; // Print index of destination neuron
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "Output Neurons (" << outputIndexStart << " to " << outputIndexStart + architectureDefinition.numberOfOutputs - 1 << ")" << std::endl;
            for (int i = outputIndexStart; i < outputIndexStart + architectureDefinition.numberOfOutputs; ++i) {
                std::cout << "  Neuron " << i - outputIndexStart << " receives input from: ";
                for (const auto& dest : neurons[i].GetSourceNeurons()) {
                    auto sourceIndex = dest - &neurons[0];
                    if(sourceIndex >= reservoirIndexStart && sourceIndex < reservoirIndexStart + architectureDefinition.numberInReservoir) {
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
        NeuralNetworkArchitectureDefinition architectureDefinition;
        int neuronCount = 0; // Count of neurons
        int inputIndexStart = 0;
        int outputIndexStart = 0;
        int reservoirIndexStart = 0;

        std::vector<float> membraneThresholds;
        std::vector<float> decayRates;  
        std::vector<float> spikeStrengths;

        void InitNeurons(bool verbose = false)
        {
            // std::cout << "|Input|Output|Reservoir|\n";
            for (int i = 0; i < architectureDefinition.numberOfInputs + architectureDefinition.numberOfOutputs + architectureDefinition.numberInReservoir; ++i)
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
                bool isReservoirNeuron = neuronCount >= reservoirIndexStart;
                
                const int spaces = neuronCount;

                std::function<void()> spikingCallback = verbose && !isReservoirNeuron ? std::function<void()>([spaces, neuronIndex, neuronType]()
                {
                    std::cout << std::string(spaces, ' ') << neuronIndex << "\t\t\t" << neuronType << " neuron spiked!\n";
                    // std::cout << "Neuron " << neuronIndex << " (" << neuronType << ") spiked!" << std::endl;
                }) : std::function<void()>([]{});

                neurons[neuronCount] = IterativeNeuron(membraneThresholds[neuronCount], decayRates[neuronCount], spikeStrengths[neuronCount], spikingCallback);
                ++neuronCount;
            }
        }

        void SetUpInputConnections()
        {
            if (architectureDefinition.numberInReservoir == 0) {
                return;
            }

            std::random_device randomDevice;
            std::mt19937 randomEngine(randomDevice());
            std::uniform_int_distribution<int> reservoirDistribution(
                architectureDefinition.numberOfInputs + architectureDefinition.numberOfOutputs,
                architectureDefinition.numberOfInputs + architectureDefinition.numberOfOutputs + architectureDefinition.numberInReservoir - 1);

            for (int inputIndex = 0; inputIndex < architectureDefinition.numberOfInputs; ++inputIndex) {
                const int reservoirIndex = reservoirDistribution(randomEngine);
                neurons[reservoirIndex].AddSynapticInput(&neurons[inputIndex + inputIndexStart]);
            }
        }

        void SetUpReservoirConnections(int max_connections_per_neuron)
        {
            if (architectureDefinition.numberInReservoir == 0) {
                return;
            }
            if (max_connections_per_neuron < 1) {
                throw std::invalid_argument("max_connections_per_neuron must be at least 1.");
            }

            std::random_device randomDevice;
            std::mt19937 randomEngine(randomDevice());
            const int reservoirStart = architectureDefinition.numberOfInputs + architectureDefinition.numberOfOutputs;
            for (int source = 0; source < architectureDefinition.numberInReservoir; ++source) {
                std::vector<int> possibleDestinations;
                possibleDestinations.reserve(architectureDefinition.numberInReservoir);
                for (int destination = 0; destination < architectureDefinition.numberInReservoir; ++destination) {
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
            if (architectureDefinition.numberInReservoir == 0) {
                return;
            }

            std::random_device randomDevice;
            std::mt19937 randomEngine(randomDevice());
            std::uniform_int_distribution<int> reservoirDistribution(
                architectureDefinition.numberOfInputs + architectureDefinition.numberOfOutputs,
                architectureDefinition.numberOfInputs + architectureDefinition.numberOfOutputs + architectureDefinition.numberInReservoir - 1);

            for (int outputIndex = 0; outputIndex < architectureDefinition.numberOfOutputs; ++outputIndex) {
                const int reservoirIndex = reservoirDistribution(randomEngine);
                neurons[outputIndex + outputIndexStart].AddSynapticInput(&neurons[reservoirIndex]);
            }
        }
};