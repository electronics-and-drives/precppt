#ifndef PCL_H__
#define PCL_H__

#include <torch/script.h>
#include <memory>

class PreceptModule
{
    private:
        torch::jit::script::Module module;
        int inputDim;
        int outputDim;

    public:
        PreceptModule(const char*, int, int);
        float* predict(const float*);
};

#endif
