#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> KHlinear_cuda_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias
);

std::vector<torch::Tensor> KHlinear_cuda_backward(
  torch::Tensor grad_output,
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> KHlinear_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return  KHlinear_cuda_forward(input, weights, bias);
}

std::vector<torch::Tensor> KHlinear_backward(
  torch::Tensor grad_output,
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias)
{
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return KHlinear_cuda_backward(grad_output, input, weights, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward", &KHlinear_forward, "KHlinear forward");
    m.def("backward", &KHlinear_backward, "KHlinear backward");
}