#include <torch/extension.h>

#include <ATen/ATen.h>

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/cuda/im2col.cuh>
#include <ATen/native/im2col_shape_check.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/im2col_native.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

torch::Tensor print_shape_cuda(torch::Tensor tensor, const std::string& name) {
    c10::ArrayRef<int64_t> shape = tensor.sizes();

    std::cout << name;
    std::cout << " Tensor shape: ";
    for (int64_t dim : shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    return tensor;
}

std::vector<torch::Tensor> KHConv2d_cuda_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor dilation,
  torch::Tensor stride,
  torch::Tensor padding
)
{
  /* Decode input */
  const int batch_size = input.size(0);
  const int input_channel = input.size(1);
  const int input_height = input.size(2);
  const int input_width = input.size(3);

  /* Decode conv2d parameters */
  const int  kernel_height = weights.size(2);
  const int  kernel_width = weights.size(3);
  const int  dilation_height = dilation.item<int>();
  const int  dilation_width = dilation.item<int>();
  const int  pad_height = padding.item<int>();
  const int  pad_width = padding.item<int>();
  const int  stride_height = stride.item<int>();
  const int  stride_width = stride.item<int>();

  at::native::im2col_shape_check(
    input,
    torch::Tensor(),
    kernel_height,
    kernel_width,
    dilation_height,
    dilation_width,
    pad_height,
    pad_width,
    stride_height,
    stride_width);

  /* Find output format */
  const int output_channel = weights.size(0);
  const int output_height = (input_height + 2 * pad_height - kernel_height) /
    dilation_height + 1;
  const int output_width = (input_width + 2 * pad_width - kernel_width) /
    dilation_width + 1;

  auto x_col = torch::zeros({batch_size, input_channel * kernel_height * kernel_width, output_height * output_width},
      torch::TensorOptions().device(torch::kCUDA));

  auto w_col = weights.reshape({output_channel,
      input_channel * kernel_height * kernel_width}).transpose(0,1);

  AT_DISPATCH_FLOATING_TYPES(at::ScalarType::Float, "KHConv2d_forward_cuda", ([&] {
        for (int i = 0; i < batch_size; i++) {
          auto x_i = input.select(0, i);
          auto x_col_i = x_col.select(0, i);

          at::native::im2col<scalar_t>(
            at::cuda::getCurrentCUDAStream(),
            x_i.const_data_ptr<scalar_t>(),
            input_channel,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            pad_height,
            pad_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            x_col_i.mutable_data_ptr<scalar_t>());
        }

    }));

  x_col = x_col.permute({0, 2, 1}).reshape({batch_size * output_height * output_width, input_channel * kernel_height * kernel_width});
  
  auto col_bias = bias.reshape({1, output_channel});
  /* TODO: Replace with custom mat mul and add */
  auto col_output = torch::addmm(bias, x_col, w_col);
  auto col_output2 = col_output.reshape({batch_size, output_height, output_width, output_channel});
  auto output = col_output2.permute({0, 3, 1, 2});

  return {output, x_col};
}

std::vector<torch::Tensor> KHConv2d_cuda_backward(
  torch::Tensor grad_output,
  torch::Tensor input,
  torch::Tensor input_col,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor dilation,
  torch::Tensor stride,
  torch::Tensor padding
)
{
  /* Decode input */
  const int batch_size = input.size(0);
  const int input_channel = input.size(1);
  const int input_height = input.size(2);
  const int input_width = input.size(3);
  const int output_channel = weights.size(0);
  const int output_height = grad_output.size(2);
  const int output_width = grad_output.size(3);

  /* Decode conv2d parameters */
  const int  kernel_height = weights.size(2);
  const int  kernel_width = weights.size(3);
  const int  dilation_height = dilation.item<int>();
  const int  dilation_width = dilation.item<int>();
  const int  pad_height = padding.item<int>();
  const int  pad_width = padding.item<int>();
  const int  stride_height = stride.item<int>();
  const int  stride_width = stride.item<int>();

  auto w_col = weights.reshape({output_channel,
      input_channel * kernel_height * kernel_width});

  auto grad_output_2d_tmp = grad_output.permute({0, 2, 3, 1});
  auto grad_output_2d = grad_output_2d_tmp.reshape({batch_size * output_height * output_width, output_channel});

  auto grad_bias = torch::sum(grad_output_2d, 0, true) / batch_size;

  /* TODO: Replace with custom mat mul */
  auto grad_w_2d = torch::mm(input_col.transpose(0, 1), grad_output_2d);

  grad_w_2d = grad_w_2d / batch_size;

  auto grad_w = grad_w_2d.transpose(0, 1).reshape({output_channel, input_channel, kernel_height, kernel_width});

  /* TODO: Replace with custom mat mul */
  auto grad_input_col = torch::mm(grad_output_2d, w_col);

  auto grad_input_col_tmp = grad_input_col.reshape({batch_size, output_height * output_width, input_channel * kernel_height * kernel_width}).permute({0, 2, 1});
  /* col2im to get grad_input */
  at::native::col2im_shape_check(
      grad_input_col_tmp,
      torch::Tensor(),
      input_height,
      input_width,
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  int64_t n_input_plane = grad_input_col_tmp.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);
  int64_t input_batch_stride = grad_input_col_tmp.stride(0);
  
  auto grad_input = torch::zeros({batch_size, input_channel, input_height, input_width},
      torch::TensorOptions().device(torch::kCUDA));
  int64_t output_batch_stride = grad_input.stride(0);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(at::ScalarType::Half, at::ScalarType::Int, at::ScalarType::Bool,
      input.scalar_type(), "col2im_out_cuda", [&] {
    int64_t height_col = (input_height + 2 * pad_height -
                          (dilation_height * (kernel_height - 1) + 1)) /
            stride_height + 1;
    int64_t width_col = (input_width + 2 * pad_width -
                         (dilation_width * (kernel_width - 1) + 1)) /
            stride_width + 1;
 
    at::native::col2im_batched(
        at::cuda::getCurrentCUDAStream(),
        grad_input_col_tmp.const_data_ptr<scalar_t>(),
        input_batch_stride,
        batch_size,
        n_output_plane,
        input_height,
        input_width,
        height_col,
        width_col,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        grad_input.mutable_data_ptr<scalar_t>(),
        output_batch_stride);

  });
  
  return {grad_input, grad_w, grad_bias};
}