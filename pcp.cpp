#include "pcp.h"

// Constructor
//  Takes path to torchscript model (*.pt) and config (*.yml) and returns an
//  inference model.
//  PreceptModule :: String -> String -> Model
PreceptModule::PreceptModule(const char* modelPath, const char* configPath)
{
    try 
        { module = torch::jit::load(modelPath); }
    catch (const c10::Error& e) 
        { std::cerr << "error loading the model\n"; }
    
    YAML::Node config = YAML::LoadFile(configPath);

    numX = config["num_x"].as<int>();
    numY = config["num_y"].as<int>();

    maxX = config["max_x"].as<std::vector<float>>();
    maxY = config["max_y"].as<std::vector<float>>();
    minX = config["min_x"].as<std::vector<float>>();
    minY = config["min_y"].as<std::vector<float>>();
    
    maskX = config["mask_x"].as<std::vector<std::string>>();
    maskY = config["mask_y"].as<std::vector<std::string>>();
    
    lambdaX = config["lambdas_x"].as<std::vector<float>>();
    lambdaY = config["lambdas_y"].as<std::vector<float>>();
    
    paramsX = config["params_x"].as<std::vector<std::string>>();
    paramsY = config["params_y"].as<std::vector<std::string>>();
}

// Predict :: [float] -> [float]
float* PreceptModule::predict(const float* x)
{
    float inputArray[numX];
    std::memcpy(inputArray, x, numX);

    at::Tensor inputTensor = torch::from_blob(inputArray, {1, numX});

    std::vector<torch::jit::IValue> input;
    input.push_back(inputTensor);

    float* y = module.forward(input).toTensor().data_ptr<float>();
    return y;
}

// Getters
int PreceptModule::getNumInputs() const {return numX;}
int PreceptModule::getNumOutputs() const {return numY;}
