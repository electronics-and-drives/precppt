#ifndef PCL_H__
#define PCL_H__

#include <torch/script.h>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <iostream>

class PreceptModule
{
    private:
        // The torchscript Module
        torch::jit::script::Module module;
        // The module configuration
        YAML::Node config;

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

        // Name of Parameters
        std::vector<std::string>paramsX;
        std::vector<std::string>paramsY;

    public:
        // Constructor
        PreceptModule(const char*, const char*);

        // Separate Loading Methods
        bool readYAMLcfg(const char*);
        bool loadTorchModel(const char*);

        // Pre-/Post-Processing Transformations
        torch::Tensor scale(const torch::Tensor, const torch::Tensor, const torch::Tensor);
        torch::Tensor unscale(const torch::Tensor, const torch::Tensor, const torch::Tensor);
        torch::Tensor boxCox(const torch::Tensor);
        torch::Tensor coxBox(const torch::Tensor);

        // Convenience Functions
        torch::Tensor scaleX(const torch::Tensor);
        torch::Tensor scaleY(const torch::Tensor);

        // Inference
        float* predict(const float*);

        // Getters
        int getNumInputs() const;
        int getNumOutputs() const;
        float* getMaxX() const;
        float* getMinX() const;
        float* getMaxY() const;
        float* getMinY() const;
        float* getLambdaX() const;
        float* getLambdaY() const;
        std::vector<std::string> getMaskX() const;
        std::vector<std::string> getMaskY() const;
        std::vector<std::string> getParamsX() const;
        std::vector<std::string> getParamsY() const;
};

#endif
