#include <torch/extension.h>
#include <iostream>

#include <vector>

std::vector<torch::Tensor> KHlinear_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias)
{
    auto output = torch::addmm(bias, input, weights.transpose(0,1));

    return {output};
}

std::vector<torch::Tensor> KHlinear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias)
{
    auto grad_input = torch::mm(grad_output, weights);
    auto grad_weights = torch::mm(grad_output.transpose(0,1), input);
    auto grad_bias = grad_output.sum(0, true);

    return {grad_input, grad_weights, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward", &KHlinear_forward, "KHlinear forward");
    m.def("backward", &KHlinear_backward, "KHlinear backward");
}