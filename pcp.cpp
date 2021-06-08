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
    
    readYAMLcfg(configPath);
}

// Read YAML Configuration generated from PRECEPT Training.
//  Takes a path to a *.yml config file and reads the nodes
//  into class attributes.
//  readYAMLcfg :: [char] -> bool
bool PreceptModule::readYAMLcfg(const char* configPath)
{
    YAML::Node config;
    try 
        { config = YAML::LoadFile(configPath); }
    catch (...) 
    { 
        std::cerr << "Error Loading " << configPath << std::endl; 
        return false;
    }

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

    return true;
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
std::vector<float> PreceptModule::getMaxX() const {return maxX;}
std::vector<float> PreceptModule::getMinX() const {return minX;}
std::vector<float> PreceptModule::getMaxY() const {return maxY;}
std::vector<float> PreceptModule::getMinY() const {return minY;}
std::vector<float> PreceptModule::getLambdaX() const {return lambdaX;}
std::vector<float> PreceptModule::getLambdaY() const {return lambdaY;}
std::vector<std::string> PreceptModule::getMaskX() const {return maskX;}
std::vector<std::string> PreceptModule::getMaskY() const {return maskY;}
std::vector<std::string> PreceptModule::getParamsX() const {return paramsX;}
std::vector<std::string> PreceptModule::getParamsY() const {return paramsY;}
