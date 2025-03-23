#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> KHConv2d_cuda_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor dilation,
  torch::Tensor stride,
  torch::Tensor padding
);

std::vector<torch::Tensor> KHConv2d_cuda_backward(
  torch::Tensor grad_output,
  torch::Tensor input,
  torch::Tensor col_input,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor dilation,
  torch::Tensor stride,
  torch::Tensor padding
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> KHConv2d_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor dilation,
  torch::Tensor stride,
  torch::Tensor padding)
{
  // CHECK_INPUT(input);
  // CHECK_INPUT(weights);
  // CHECK_INPUT(bias);

  return  KHConv2d_cuda_forward(input, weights, bias, dilation, stride, padding);
}

std::vector<torch::Tensor> KHConv2d_backward(
  torch::Tensor grad_output,
  torch::Tensor input,
  torch::Tensor col_input,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor dilation,
  torch::Tensor stride,
  torch::Tensor padding)
{
  // CHECK_INPUT(grad_output);
  // CHECK_INPUT(input);
  // CHECK_INPUT(col_input);
  // CHECK_INPUT(weights);
  // CHECK_INPUT(bias);

  return KHConv2d_cuda_backward(grad_output, input, col_input, weights, bias, dilation, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward", &KHConv2d_forward, "KHConv2d forward");
    m.def("backward", &KHConv2d_backward, "KHConv2d backward");
}