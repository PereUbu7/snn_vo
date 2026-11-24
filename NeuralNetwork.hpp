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

        void SetUpConnections(int max_connections_per_neuron, bool verbose = false) {
            InitNeurons(verbose);
            SetUpInputConnections();
            SetUpReservoirConnections(max_connections_per_neuron);
            SetUpOutputConnections();
        }

        void SetUpConnections(const std::vector<std::vector<int>>& inputConnections, const std::vector<std::vector<int>>& reservoirConnections, const std::vector<std::vector<int>>& outputConnections, bool verbose = false)
        {
            InitNeurons(verbose);

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

        void InitNeurons(bool verbose = false)
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
                bool isReservoirNeuron = neuronCount >= reservoirIndexStart;
                
                const int spaces = neuronCount;

                std::function<void()> spikingCallback = verbose && !isReservoirNeuron ? std::function<void()>([spaces, neuronIndex, neuronType]()
                {
                    std::cout << std::string(spaces, ' ') << neuronIndex << "\t\t\t" << neuronType << " neuron spiked!\n";
                    // std::cout << "Neuron " << neuronIndex << " (" << neuronType << ") spiked!" << std::endl;
                }) : std::function<void()>([]{});

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