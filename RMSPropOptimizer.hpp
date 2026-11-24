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