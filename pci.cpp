#include <iostream>
#include "pcp.h"

int main(int argc, const char* argv[]) 
{
    if(argc != 3) 
    {
        std::cerr << "usage: " << argv[0] << "  <path-to-model.pt> <path-to-config.yml>" << '\n';
        return 1;
    }

    srand(666);

    PreceptModule* model = new PreceptModule(argv[1], argv[2]);

    int inputDim = model->getNumInputs();
    int outputDim = model->getNumOutputs();

    std::cout << "TorchScript Model with" << std::endl;
    std::cout << "Num Inputs: " << inputDim << std::endl;
    std::cout << "Num Outputs: " << outputDim << std::endl;

    std::vector<float> X({ 1.5e-6, 1.5e-7, 0.6, 0.6, 0.0 });
    //std::vector<float> X;
    //for(int i = 0; i < inputDim; i++)
    //    {X.push_back(rand());}

    std::cout << "Input: [ ";
    for(float x : X)
        {std::cout << x << ", " ;}
    std::cout << "]" << std::endl;

    std::vector<float> Y = model->predict(std::vector<float>(X));

    std::cout << "Output After: [ ";
    for(int i = 0; i < outputDim; i++)
        {std::cout << Y[i] << ", " ;}
    std::cout << "]" << std::endl;

    //delete y;
    //delete model;

    return 0;
}
