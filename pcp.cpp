#include "pcp.h"

PreceptModule::PreceptModule(const char* path, int inp, int outp)
{
    try 
        { module = torch::jit::load(path); }
    catch (const c10::Error& e) 
        { std::cerr << "error loading the model\n"; }

    inputDim = inp;
    outputDim = outp;
}

float* PreceptModule::predict(const float* x)
{
    float inp[inputDim];
    std::memcpy(inp, x, inputDim);

    at::Tensor inpTT = torch::from_blob(inp, {1, inputDim});
    std::cout << inpTT.sizes() << '\n';

    at::Tensor inpT = torch::ones({1, 5});
    std::cout << inpT.sizes() << '\n';

    std::vector<torch::jit::IValue> input;
    //input.push_back(X);
    //input.push_back(torch::ones({1, 5}));
    input.push_back(inpTT);

    float* y = module.forward(input).toTensor().data_ptr<float>();

    return y;
}
