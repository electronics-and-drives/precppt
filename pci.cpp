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

    int inputDim = 5;
    int outputDim = 13;

    PreceptModule* model = new PreceptModule(argv[1], argv[2]);

    //float X[inputDim];
    //for(int i = 0; i < inputDim; i++)
    //    {X[i] = rand();}

    //float* Y = model->predict(X);

    //std::cout << "Input: [ ";
    //for(float x : X)
    //    {std::cout << x << ", " ;}
    //std::cout << "]\n";

    //std::cout << "Output: [ ";
    //for(int i = 0; i < outputDim; i++)
    //    {std::cout << Y[i] << ", " ;}
    //std::cout << "]\n";

    //delete y;
    //delete model;

    return 0;
}
