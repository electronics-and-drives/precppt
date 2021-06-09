#include "pcp.h"

// Constructor
//  Takes path to torchscript model (*.pt) and config (*.yml) and returns an
//  inference model.
//  PreceptModule :: String -> String -> Model
PreceptModule::PreceptModule(const char* modelPath, const char* configPath)
{
    loadTorchModel(modelPath);
    readYAMLcfg(configPath);
}

// Load TorchScript Model generated from PRECEPT Training.
//  Takes a path to a *.pt model file and loads it.
//  loadTorchModel :: [char] -> bool
bool PreceptModule::loadTorchModel(const char* modelPath)
{
    try 
        { module = torch::jit::load(modelPath); }
    catch (const c10::Error& e) 
    { 
        std::cerr << "Error Loading " << modelPath << std::endl; 
        return false;
    }

    return true;
}

// Read YAML Configuration generated from PRECEPT Training.
//  Takes a path to a *.yml config file and reads the nodes
//  into class attributes.
//  readYAMLcfg :: [char] -> bool
bool PreceptModule::readYAMLcfg(const char* configPath)
{
    try 
        { config = YAML::LoadFile(configPath); }
    catch (...) 
    { 
        std::cerr << "Error Loading " << configPath << std::endl; 
        return false;
    }

    paramsX = config["params_x"].as<std::vector<std::string>>();
    paramsY = config["params_y"].as<std::vector<std::string>>();

    numX = config["num_x"].as<int>();
    numY = config["num_y"].as<int>();
    
    maxX = torch::from_blob(config["max_x"].as<std::vector<float>>().data(), {1, numX});
    maxY = torch::from_blob(config["max_y"].as<std::vector<float>>().data(), {1, numY});
    minX = torch::from_blob(config["min_x"].as<std::vector<float>>().data(), {1, numX});
    minY = torch::from_blob(config["min_y"].as<std::vector<float>>().data(), {1, numY});

    maskX = config["mask_x"].as<std::vector<std::string>>();
    maskY = config["mask_y"].as<std::vector<std::string>>();
    
    //lambdaX = torch::from_blob(config["lambdas_x"].as<std::vector<float>>());
    //lambdaY = torch::from_blob(config["lambdas_y"].as<std::vector<float>>());
    
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
//float* PreceptModule::getMaxX() const {return maxX.data_ptr<float>();}
//float* PreceptModule::getMinX() const {return minX.data_ptr<float>();}
//float* PreceptModule::getMaxY() const {return maxY.data_ptr<float>();}
//float* PreceptModule::getMinY() const {return minY.data_ptr<float>();}
//float* PreceptModule::getLambdaX() const {return lambdaX.data_ptr<float>();}
//float* PreceptModule::getLambdaY() const {return lambdaY.data_ptr<float>();}
//std::vector<std::string> PreceptModule::getMaskX() const {return maskX;}
//std::vector<std::string> PreceptModule::getMaskY() const {return maskY;}
//std::vector<std::string> PreceptModule::getParamsX() const {return paramsX;}
//std::vector<std::string> PreceptModule::getParamsY() const {return paramsY;}
