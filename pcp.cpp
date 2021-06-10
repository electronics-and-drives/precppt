#include "pcp.h"

// Constructor
//  Takes path to torchscript model (*.pt) and config (*.yml) and returns an
//  inference model.
//  PreceptModule :: String -> String -> Model
PreceptModule::PreceptModule(const char* mdlPath, const char* cfgPath)
{
    std::string modelPath(mdlPath);
    std::string configPath(cfgPath);

    loadTorchModel(modelPath);
    readYAMLcfg(configPath);
}

// Utility functions for converting between std::vector and torch::tensor for
// interfacing with other applications.
// t2v :: Tensor -> vector
std::vector<float> PreceptModule::ten2vec(torch::Tensor t) const
{ 
    return std::vector<float>( t.data_ptr<float>() 
                             , (t.data_ptr<float>() + t.numel())); 
}
// v2t :: vector -> Tensor
torch::Tensor PreceptModule::vec2ten(const std::vector<float> v) const
{ 
    return torch::tensor(v, defaultOptions)
                 .reshape({static_cast<long>(v.size()),1})
                 .clone(); 
}
    

// Load TorchScript Model generated from PRECEPT Training.
//  Takes a path to a *.pt model file and loads it.
//  loadTorchModel :: [char] -> bool
bool PreceptModule::loadTorchModel(const std::string modelPath)
{
    try 
    { 
        module = torch::jit::load(modelPath); 
    }
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
bool PreceptModule::readYAMLcfg(const std::string configPath)
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
    maxX = vec2ten(config["max_x"].as<std::vector<float>>());
    minX = vec2ten(config["min_x"].as<std::vector<float>>());
    maxY = vec2ten(config["max_y"].as<std::vector<float>>());
    minY = vec2ten(config["min_y"].as<std::vector<float>>());

    // Transformation Mask
    maskX = config["mask_x"].as<std::vector<std::string>>();
    maskY = config["mask_y"].as<std::vector<std::string>>();

    // Transformation mask as index
    std::for_each( maskX.begin(), maskX.end()
                 , [this] (std::string m) -> void
                 { 
                    maskXidx.push_back( std::distance( paramsX.begin()
                                                     , std::find( paramsX.begin()
                                                                , paramsX.end()
                                                                , m ))); 
                 });

    std::for_each( maskY.begin(), maskY.end()
                 , [this] (std::string m) -> void
                 { 
                    maskYidx.push_back( std::distance( paramsY.begin()
                                                     , std::find( paramsY.begin()
                                                                , paramsY.end()
                                                                , m ))); 
                 });

    // Lambda parameter for Box-Cox transformation
    lambdaX = vec2ten(config["lambdas_x"].as<std::vector<float>>());
    lambdaY = vec2ten(config["lambdas_y"].as<std::vector<float>>());

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
torch::Tensor PreceptModule::scale( const torch::Tensor var
                                  , const torch::Tensor min
                                  , const torch::Tensor max )
    { return ((var - min) / (max - min)); }


// Box-Cox transformation
//
// For λ ≠ 0:
// 
//       λ  
//      y -1
//   y'=――――
//       λ  
//       
// otherwise:
// 
//   y'=ln(y)
//
//  boxCox :: Tensor -> Tensor
torch::Tensor PreceptModule::boxCox(const torch::Tensor var, const float lambda)
{ 
    if(lambda != 0.0)
        { return ((var.pow(lambda) - 1.0) / lambda); }
    else
        { return var.log(); }
}

// Inverse Box-Cox Transformation
//
// For λ ≠ 0:
//
//      ⎛ln(y'∙λ+1)⎞
//      ⎜――――――――――⎟
//      ⎝    λ     ⎠
//   y=e            
// otherwise:
//
//      y'
//   y=e  
// coxBox :: Tensor -> Tensor
torch::Tensor PreceptModule::coxBox(const torch::Tensor var, const float lambda)
{
    if(lambda != 0.0)
        { return torch::exp( torch::log( var * lambda + 1 ) / lambda ); }
    else
        { return torch::exp(var); }
}

// Un-scale the output of the model back to raw/real data
// according to the transformation used during training.
// Takes a Tensor, and returns one of the same size with values scaled [0;1]
// based on the minima and maxima specified in the config.
//
//  x = x' ∙ (max(x) - min(x)) + min(x)
//
//  scale :: Tensor -> Tensor
torch::Tensor PreceptModule::unscale( const torch::Tensor var
                                    , const torch::Tensor min
                                    , const torch::Tensor max )
    { return (var * (max - min) + min); }

// Conveniece wrappers for scaling
torch::Tensor PreceptModule::scaleX(const torch::Tensor X) 
    { return scale(X, minX, maxX); }
torch::Tensor PreceptModule::scaleY(const torch::Tensor Y) 
    { return unscale(Y, minY, maxY); }

// Evaluate the model for a given input. Takes a float vector of raw data
// corresponding to the form specified in the config (*.yml).
//  predict :: [float] -> [float]
std::vector<float> PreceptModule::predict(const std::vector<float> x)
{
    // Turn input std::vector into torch::Tensor
    torch::Tensor X = vec2ten(x).reshape({numX,1});

    // Transform inputs
    torch::Tensor trafoX = X;
    if(!maskX.empty())
    {  
        int lIdx = 0;
        std::for_each( maskXidx.begin(), maskXidx.end()
                     , [this, &trafoX, &lIdx] (long mIdx) -> void
                     {
                        trafoX[mIdx][0] = coxBox( trafoX[mIdx][0]
                                                , lambdaX[lIdx].item().toFloat());
                        lIdx++;
                     });
    }

    // Scale transformed inputs [0;1]
    torch::Tensor scaledX = scaleX(trafoX);

    // Push into batch
    std::vector<torch::jit::IValue> input;
    input.push_back(scaledX.transpose(0,1));

    // Forward pass through network
    torch::Tensor scaledY = torch::transpose( module.forward(input)
                                                    .toTensor()
                                                    .to(defaultOptions)
                                            , 0, 1);

    // Unscale output
    torch::Tensor trafoY = scaleY(scaledY).to(defaultOptions);

    // Transform outputs
    torch::Tensor Y = trafoY;
    if(!maskY.empty())
    {  
        int lIdx = 0;
        std::for_each( maskYidx.begin(), maskYidx.end()
                     , [this, &Y, &lIdx] (long mIdx) -> void
                     {
                        Y[mIdx][0] = coxBox( Y[mIdx][0]
                                           , lambdaY[lIdx].item().toFloat());
                        lIdx++;
                     });
    }

    // Convert output to std::vector
    std::vector<float> y = ten2vec(Y);

    return y;
}

// Getters
int PreceptModule::getNumInputs() const {return numX;}
int PreceptModule::getNumOutputs() const {return numY;}
std::vector<float> PreceptModule::getMaxX() const { return ten2vec(maxX); }
std::vector<float> PreceptModule::getMinX() const { return ten2vec(minX); }
std::vector<float> PreceptModule::getMaxY() const {return ten2vec(maxY);}
std::vector<float> PreceptModule::getMinY() const {return ten2vec(minY);}
std::vector<float> PreceptModule::getLambdaX() const {return ten2vec(lambdaX);}
std::vector<float> PreceptModule::getLambdaY() const {return ten2vec(lambdaY);}
std::vector<std::string> PreceptModule::getMaskX() const {return maskX;}
std::vector<std::string> PreceptModule::getMaskY() const {return maskY;}
std::vector<std::string> PreceptModule::getParamsX() const {return paramsX;}
std::vector<std::string> PreceptModule::getParamsY() const {return paramsY;}
