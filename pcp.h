#ifndef PCL_H__
#define PCL_H__

#include <torch/script.h>
#include <yaml-cpp/yaml.h>
#include <memory>

class PreceptModule
{
    private:
        // The torchscript Module
        torch::jit::script::Module module;

        // Number of neurons at the input, this is how long the input array
        // needs to be.
        int numX;
        // Number of neurons at the output, this is how long the response
        // (prediction) array will be.
        int numY;

        // Minima and Maxima of training data, this is used to (re-)scale the
        // inputs and outptus
        std::vector<float>maxX;
        std::vector<float>minX;
        std::vector<float>maxY;
        std::vector<float>minY;
       
        // Box-Cox Transformation Mask
        std::vector<std::string>maskX;
        std::vector<std::string>maskY;

        // Lambdas used for Transformation
        std::vector<float>lambdaX;
        std::vector<float>lambdaY;

        // Name of Parameters
        std::vector<std::string>paramsX;
        std::vector<std::string>paramsY;

    public:
        // Constructor
        PreceptModule(const char*, const char*);

        // Separate Loading Methods
        bool readYAMLcfg(const char*);

        // Inference
        float* predict(const float*);

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
