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

            optimizer.ApplyGradients(
                membraneThresholdMus, membraneThresholdSigmas, membraneMuGradient, membraneSigmaGradient,
                decayRateMus, decayRateSigmas, decayRateMuGradient, decayRateSigmaGradient,
                spikeStrengthMus, spikeStrengthSigmas, spikeStrengthMuGradient, spikeStrengthSigmaGradient);
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