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