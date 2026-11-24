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