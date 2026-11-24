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