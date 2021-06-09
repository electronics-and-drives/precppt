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

    // Column names of in- and outputs
    paramsX = config["params_x"].as<std::vector<std::string>>();
    paramsY = config["params_y"].as<std::vector<std::string>>();

    // Number of in- and outputs
    numX = config["num_x"].as<int>();
    numY = config["num_y"].as<int>();

    // Minima and Maxima for scaling / normalizing
    auto rawMaxX = config["max_x"].as<std::vector<float>>();
    auto rawMaxY = config["max_y"].as<std::vector<float>>();
    auto rawMinX = config["min_x"].as<std::vector<float>>();
    auto rawMinY = config["min_y"].as<std::vector<float>>();

    maxX = torch::from_blob(rawMaxX.data(), {1, numX});
    maxY = torch::from_blob(rawMaxY.data(), {1, numY});
    minX = torch::from_blob(rawMinX.data(), {1, numX});
    minY = torch::from_blob(rawMinY.data(), {1, numY});

    // Transformation Mask
    maskX = config["mask_x"].as<std::vector<std::string>>();
    maskY = config["mask_y"].as<std::vector<std::string>>();
    
    // Lambda parameter for Box-Cox transformation
    auto rawLX = config["lambdas_x"].as<std::vector<float>>();
    auto rawLY = config["lambdas_y"].as<std::vector<float>>();

    lambdaX = torch::from_blob( rawLX.data()
                              , {1, static_cast<long>(maskX.size())} );
    lambdaY = torch::from_blob( rawLY.data()
                              , {1, static_cast<long>(maskY.size())} );
    
    return true;
}

// Scale the input data according to the transformation used during training.
// Takes a Tensor, and returns one of the same size with values scaled [0;1]
// based on the minima and maxima specified in the config.
//
//       ⎛  x-min(x)   ⎞
//  x' = ⎜―――――――――――――⎟
//       ⎝max(x)-min(x)⎠
//
//  scale :: Tensor -> Tensor
at::Tensor PreceptModule::scale( const at::Tensor var
                               , const at::Tensor min
                               , const at::Tensor max )
    { return ((var - min) / (max - min)); }

// Un-scale the output of the model back to raw/real data
// according to the transformation used during training.
// Takes a Tensor, and returns one of the same size with values scaled [0;1]
// based on the minima and maxima specified in the config.
//
//  x = x' ∙ (max(x) - min(x)) + min(x)
//
//  scale :: Tensor -> Tensor
at::Tensor PreceptModule::unscale( const at::Tensor var
                                 , const at::Tensor min
                                 , const at::Tensor max )
    { return (var * (max - min) + min); }

// Conveniece wrappers for scaling
at::Tensor PreceptModule::scaleX(const at::Tensor X) 
    { return scale(X, minX, maxX); }
at::Tensor PreceptModule::scaleY(const at::Tensor Y) 
    { return unscale(Y, minY, maxY); }

// Evaluate the model for a given input. Takes a float array of raw data
// corresponding to the form specified in the config (*.yml).
//  predict :: [float] -> [float]
float* PreceptModule::predict(const float* x)
{
    float inputArray[numX];
    std::memcpy(inputArray, x, numX);

    at::Tensor X = torch::from_blob(inputArray, {1, numX});

    std::cout << "X: ";
    std::cout << X.slice(1, 0, numX) << std::endl;
    at::Tensor scaledX = scaleX(X);
    std::cout << "Scaled X: ";
    std::cout << scaledX.slice(1, 0, 5) << std::endl;

    std::vector<torch::jit::IValue> input;
    input.push_back(scaledX);
    at::Tensor scaledY = module.forward(input).toTensor();

    at::Tensor Y = scaleY(scaledY);
    float* y = Y.data_ptr<float>();
    //float* y = module.forward(input).toTensor().data_ptr<float>();

    return y;
}

// Getters
int PreceptModule::getNumInputs() const {return numX;}
int PreceptModule::getNumOutputs() const {return numY;}
float* PreceptModule::getMaxX() const {return maxX.data_ptr<float>();}
float* PreceptModule::getMinX() const {return minX.data_ptr<float>();}
float* PreceptModule::getMaxY() const {return maxY.data_ptr<float>();}
float* PreceptModule::getMinY() const {return minY.data_ptr<float>();}
float* PreceptModule::getLambdaX() const {return lambdaX.data_ptr<float>();}
float* PreceptModule::getLambdaY() const {return lambdaY.data_ptr<float>();}
std::vector<std::string> PreceptModule::getMaskX() const {return maskX;}
std::vector<std::string> PreceptModule::getMaskY() const {return maskY;}
std::vector<std::string> PreceptModule::getParamsX() const {return paramsX;}
std::vector<std::string> PreceptModule::getParamsY() const {return paramsY;}
