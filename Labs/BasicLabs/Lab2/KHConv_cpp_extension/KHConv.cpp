#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <vector>

torch::Tensor Img2Col(
    torch::Tensor input,
    torch::Tensor weights,
    int64_t stride,
    int64_t padding,
    int64_t out_height,
    int64_t out_width)
{
    int64_t batch_size = input.size(0);
    int64_t in_channel = weights.size(1);
    int64_t kernel_height = weights.size(2);
    int64_t kernel_width = weights.size(3);

     /* 
      * im2col out dim: (batch_size) * (in_channel * kernel_height * kernel_width) * (out_height * out_width)
      * => need transpose => dim: (batch_size) *(out_height * out_width) * (in_channel * kernel_height * kernel_width)
      */
    torch::Tensor col_tmp = torch::im2col(input.clone(),
        torch::IntArrayRef({kernel_height, kernel_width}), /*kernel_size*/
        torch::IntArrayRef({1, 1}), /*dilation*/
        torch::IntArrayRef({padding, padding}), /*padding*/
        torch::IntArrayRef({stride, stride}) /*stride*/).permute(torch::IntArrayRef({0, 2, 1}));

    torch::Tensor col = col_tmp.reshape(torch::IntArrayRef({batch_size * out_height * out_width,
        in_channel * kernel_height * kernel_width}));

    return {col};
}

std::vector<torch::Tensor> KHConv_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor stride,
    torch::Tensor padding)
{
    int64_t kernel_height = weights.size(2);
    int64_t kernel_width = weights.size(3);
    int64_t padding_val = padding.item<int>();
    int64_t stride_val = stride.item<int>();
    int64_t batch_size = input.size(0);
    int64_t out_channel = weights.size(0);
    int64_t in_channel = weights.size(1);
    int64_t out_height = ((input.size(2) - kernel_height + 2 * padding_val) / stride_val) + 1;
    int64_t out_width = ((input.size(3) - kernel_width + 2 * padding_val) / stride_val) + 1;

    /* col_w dim = (in_chanel * kernel_height * kernel_width) * out_channel */
    torch::Tensor col_w = weights.reshape(torch::IntArrayRef({out_channel,
        in_channel * kernel_height * kernel_width})).transpose(0,1);

    /* col_b dim = 1 * out_channel */
    torch::Tensor col_b = bias.reshape(torch::IntArrayRef({1, out_channel}));

    /* im2col columns.dim: (batchSize * out_height * out_width) * (in_channel * kernel_height * kernel_width) */
    torch::Tensor col_input = Img2Col(input, weights, stride_val, padding_val, out_height, out_width);

    torch::Tensor col_output = torch::addmm(col_b, col_input, col_w);
    torch::Tensor col_output2 = col_output.reshape(torch::IntArrayRef({batch_size, out_height, out_width, out_channel}));
    torch::Tensor output = col_output2.permute(torch::IntArrayRef({0, 3, 1, 2}));

    return {output, col_input};
}

torch::Tensor Col2Img(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor col_w,
    int64_t stride,
    int64_t padding,
    int64_t out_height,
    int64_t out_width)
{
    int64_t batch_size = input.size(0);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);
    int64_t in_channel = weights.size(1);
    int64_t kernel_height = weights.size(2);
    int64_t kernel_width = weights.size(3);

    /* dim = batch_size * (out_height * out_width) * (in_channel * kernel_height * kernel_width) */
    torch::Tensor col_tmp = col_w.reshape(torch::IntArrayRef({batch_size, out_height * out_width,
        in_channel * kernel_height * kernel_width}));

    /* dim = batch_size * (in_channel * kernel_height * kernel_width) * (out_height * out_width) */
    torch::Tensor col_tmp2 = col_tmp.permute(torch::IntArrayRef({0, 2, 1}));

     /* 
      * col2im out dim: batch_size * in_channel * in_height * in_width
      */
    torch::Tensor img = torch::col2im(col_tmp2.clone(),
        torch::IntArrayRef({in_height, in_width}), /*output_size*/
        torch::IntArrayRef({kernel_height, kernel_width}), /*kernel_size*/
        torch::IntArrayRef({1, 1}), /*dilation*/
        torch::IntArrayRef({padding, padding}), /*padding*/
        torch::IntArrayRef({stride, stride}) /*stride*/);

    return {img};
}

std::vector<torch::Tensor> KHConv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor col_input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor stride,
    torch::Tensor padding)
{
    int64_t kernel_height = weights.size(2);
    int64_t kernel_width = weights.size(3);
    int64_t padding_val = padding.item<int>();
    int64_t stride_val = stride.item<int>();

    int64_t batch_size = input.size(0);
    int64_t out_channel = weights.size(0);
    int64_t in_channel = weights.size(1);
    int64_t out_height = grad_output.size(2);
    int64_t out_width = grad_output.size(3);

    /* col_w dim = (in_chanel * kernel_height * kernel_width) * out_channel */
    torch::Tensor col_w = weights.reshape(torch::IntArrayRef({out_channel,
        in_channel * kernel_height * kernel_width})).transpose(0,1);

    /* dim = (batch_size * out_height * out_width) * out_channel */
    auto grad_output_2d = grad_output.permute(torch::IntArrayRef({0, 2, 3, 1}))\
        .reshape(torch::IntArrayRef({batch_size * out_height * out_width, out_channel}));

    auto grad_bias = torch::sum(grad_output_2d, 0, true) / batch_size;

    /* dim = (in_channel * kernel_height * kernel_width) * out_channel */
    auto grad_weights_2d = torch::mm(col_input.transpose(0,1), grad_output_2d) / batch_size;
    /* dim = out_channel * in_channel * kernel_height * kernel_width */
    auto grad_weights = grad_weights_2d.transpose(0,1)
        .reshape(torch::IntArrayRef({out_channel, in_channel, kernel_height, kernel_width}));

    /* dim = (batch_size * out_height * out_width) * (in_channel * kernel_height * kernel_width) */
    auto grad_input_col = torch::mm(grad_output_2d, col_w.transpose(0,1));

    auto grad_input = Col2Img(input, weights, grad_input_col, stride_val, padding_val, out_height, out_width);

    return {grad_input, grad_weights, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward", &KHConv_forward, "KHConv forward");
    m.def("backward", &KHConv_backward, "KHConv backward");
}

