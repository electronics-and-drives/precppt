#ifndef PCL_H__
#define PCL_H__

#include <torch/script.h>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <iostream>

class PreceptModule
{
    private:
        // Maintain path names as strings
        std::string modelPath;
        std::string configPath;

        // The torchscript Module
        torch::jit::script::Module module;
        // The module configuration
        YAML::Node config;
        // Default Tensor Options
        torch::TensorOptions defaultOptions = torch::TensorOptions()
                                                    .dtype(torch::kFloat32);

        // Number of neurons at the input, this is how long the input array
        // needs to be.
        int numX;
        // Number of neurons at the output, this is how long the response
        // (prediction) array will be.
        int numY;

        // Minima and Maxima of training data, this is used to (re-)scale the
        // inputs and outptus
        torch::Tensor maxX;
        torch::Tensor minX;
        torch::Tensor maxY;
        torch::Tensor minY;
       
        // Lambdas used for Transformation
        torch::Tensor lambdaX;
        torch::Tensor lambdaY;

        // Box-Cox Transformation Mask
        std::vector<std::string>maskX;
        std::vector<std::string>maskY;
        std::vector<long> maskXidx;
        std::vector<long> maskYidx;
        //std::vector<torch::indexing::TensorIndex> maskXidx;
        //std::vector<torch::indexing::TensorIndex> maskYidx;

        // Name of Parameters
        std::vector<std::string>paramsX;
        std::vector<std::string>paramsY;

    public:
        // Utility
        std::vector<float> ten2vec(const torch::Tensor) const;
        torch::Tensor vec2ten(const std::vector<float>) const;
        //std::vector<float> ten2vec(const torch::Tensor);
        //torch::Tensor vec2ten(const std::vector<float>);

        // Constructor
        PreceptModule(const char*, const char*);

        // Separate Loading Methods
        bool readYAMLcfg(const std::string);
        bool loadTorchModel(const std::string);

        // Pre-/Post-Processing Transformations
        torch::Tensor scale(const torch::Tensor, const torch::Tensor, const torch::Tensor);
        torch::Tensor unscale(const torch::Tensor, const torch::Tensor, const torch::Tensor);
        torch::Tensor boxCox(const torch::Tensor, const float lambda);
        torch::Tensor coxBox(const torch::Tensor, const float lambda);

        // Convenience Functions
        torch::Tensor scaleX(const torch::Tensor);
        torch::Tensor scaleY(const torch::Tensor);

        // Inference
        std::vector<float> predict(const std::vector<float>);

        // Getters
        int getNumInputs() const;
        int getNumOutputs() const;
        std::vector<float> getMaxX() const;
        std::vector<float> getMinX() const;
        std::vector<float> getMaxY() const;
        std::vector<float> getMinY() const;
        std::vector<float> getLambdaX() const;
        std::vector<float> getLambdaY() const;
        std::vector<std::string> getMaskX() const;
        std::vector<std::string> getMaskY() const;
        std::vector<std::string> getParamsX() const;
        std::vector<std::string> getParamsY() const;
};

#endif
