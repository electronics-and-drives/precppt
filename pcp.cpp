#include "pcp.h"

PreceptModule::PreceptModule(const char* modelPath, const char* configPath)
{
    try 
        { module = torch::jit::load(modelPath); }
    catch (const c10::Error& e) 
        { std::cerr << "error loading the model\n"; }
    
    YAML::Node config = YAML::LoadFile(configPath);
}

float* PreceptModule::predict(const float* x)
{
    float y = 0.0;
    //float inputArray[inputDim];
    //std::memcpy(inpArray, x, inputDim);

    //at::Tensor inputTensor = torch::from_blob(inpArray, {1, inputDim});

    //std::vector<torch::jit::IValue> input;
    //input.push_back(unputTensor);

    //float* y = module.forward(input).toTensor().data_ptr<float>();

    return &y;
}
