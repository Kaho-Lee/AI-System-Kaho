#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void kh_matmul_kernel(
  const scalar_t* X,
  const scalar_t* W,
  scalar_t* Y,
  const int Y_row, 
  const int numOfAdd, 
  const int Y_col,
  const bool X_transpose = false,
  const bool Y_transpose = false)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < Y_row && col < Y_col) {
    int row_idx, col_idx;
    scalar_t tmp = 0.0;
    for (int i = 0; i < numOfAdd; i++) {
      row_idx = X_transpose? i * Y_row + row : row * numOfAdd + i;
      col_idx = Y_transpose? col * numOfAdd + i : i * Y_col + col;
      tmp += X[row_idx] * W[col_idx];
    }
    Y[row * Y_col + col] = tmp;
  }
}

template <typename scalar_t>
__global__ void kh_matmulAdd_kernel(
  const scalar_t* X,
  const scalar_t* W,
  const scalar_t* B,
  scalar_t* Y,
  const int Y_row, 
  const int numOfAdd, 
  const int Y_col,
  const bool X_transpose = false,
  const bool Y_transpose = false)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < Y_row && col < Y_col) {
    int row_idx, col_idx;
    scalar_t tmp = 0.0;
    for (int i = 0; i < numOfAdd; i++) {
      row_idx = X_transpose? i * Y_row + row : row * numOfAdd + i;
      col_idx = Y_transpose? col * numOfAdd + i : i * Y_col + col;
      tmp += X[row_idx] * W[col_idx];
    }
    Y[row * Y_col + col] = tmp + B[col];
  }

}

std::vector<torch::Tensor> KHlinear_cuda_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias
)
{

  const int batch_size = input.size(0);
  const int input_dim = input.size(1);
  const int output_dim = weights.size(0);

  auto output = torch::zeros({batch_size, output_dim},
      torch::TensorOptions().device(torch::kCUDA));

  const int blockSize = 32;
  dim3 threadsPerBlock(blockSize, blockSize);

  dim3 blocksPerGrid(1,1);
  blocksPerGrid.x = ceil(double(batch_size) / double(blockSize));
  blocksPerGrid.y = ceil(double(output_dim) / double(blockSize));

  AT_DISPATCH_FLOATING_TYPES(output.type(), "KHlinear_forward_cuda", ([&] {
        kh_matmulAdd_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
            input.data<scalar_t>(),
            weights.data<scalar_t>(),
            bias.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            input_dim,
            output_dim,
            false,
            true);
        }));
  
   return {output};
}

std::vector<torch::Tensor> KHlinear_cuda_backward(
  torch::Tensor grad_output,
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias
)
{
  const int batch_size = grad_output.size(0);
  const int input_dim = weights.size(1);
  const int output_dim = grad_output.size(1);

  auto grad_input = torch::zeros({batch_size, input_dim},
      torch::TensorOptions().device(torch::kCUDA));

  auto grad_weights = torch::zeros({output_dim, input_dim},
      torch::TensorOptions().device(torch::kCUDA));

  const int blockSize = 32;

  const dim3 threadsPerBlock(blockSize, blockSize);

  dim3 blocksPerGrid_dInput(1,1);
  dim3 blocksPerGrid_dWeight(1,1);
  blocksPerGrid_dInput.x = ceil(double(batch_size) / double(blockSize));
  blocksPerGrid_dInput.y = ceil(double(input_dim) / double(blockSize));
  blocksPerGrid_dWeight.x = ceil(double(output_dim) / double(blockSize));
  blocksPerGrid_dWeight.y = ceil(double(input_dim) / double(blockSize));

  AT_DISPATCH_FLOATING_TYPES(input.type(), "KHlinear_cuda_backward", ([&] {
        kh_matmul_kernel<scalar_t><<<blocksPerGrid_dInput, threadsPerBlock>>>(
            grad_output.data<scalar_t>(),
            weights.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            batch_size,
            output_dim,
            input_dim,
            false,
            false);
        }));

  AT_DISPATCH_FLOATING_TYPES(input.type(), "KHlinear_cuda_backward", ([&] {
      kh_matmul_kernel<scalar_t><<<blocksPerGrid_dWeight, threadsPerBlock>>>(
          grad_output.data<scalar_t>(),
          input.data<scalar_t>(),
          grad_weights.data<scalar_t>(),
          output_dim,
          batch_size,
          input_dim,
          true,
          false);
      }));
  auto grad_bias = grad_output.sum(0, true);
  return {grad_input, grad_weights, grad_bias};
}