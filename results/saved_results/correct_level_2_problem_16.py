# runtime: 49.2
# basline: 4.5
# speedup: 0.09146341463414634
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
custom_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// More accurate Mish activation matching PyTorch
__device__ float mish(float x) {
    float e = expf(x);
    float n = e * e + 2 * e;
    if (x <= -0.6f) {
        return x * (n / (n + 2));
    }
    return x * tanhf(log1pf(e));
}

// Corrected conv_transpose + Mish kernel
__global__ void conv_transpose_mish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding) {
    
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (c >= out_channels || h >= out_height || w >= out_width) {
        return;
    }

    for (int n = 0; n < batch_size; ++n) {
        float val = bias != nullptr ? bias[c] : 0.0f;

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_h = (h - kh + padding) / stride;
                    int input_w = (w - kw + padding) / stride;
                    
                    if ((h - kh + padding) % stride == 0 && 
                        (w - kw + padding) % stride == 0 &&
                        input_h >= 0 && input_h < in_height &&
                        input_w >= 0 && input_w < in_width) {
                        
                        int input_idx = n * in_channels * in_height * in_width + 
                                     ic * in_height * in_width + 
                                     input_h * in_width + 
                                     input_w;
                        
                        int weight_idx = ic * out_channels * kernel_size * kernel_size + 
                                      c * kernel_size * kernel_size + 
                                      kh * kernel_size + 
                                      kw;
                        
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        int output_idx = n * out_channels * out_height * out_width + 
                       c * out_height * out_width + 
                       h * out_width + 
                       w;
        output[output_idx] = mish(val);
    }
}

// More numerically stable add + hardtanh + scale
__global__ void add_hardtanh_scale_kernel(
    float* input_output,
    float add_value,
    float scale,
    int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    float val = input_output[idx] + add_value;
    val = val > 1.0f ? 1.0f : (val < -1.0f ? -1.0f : val);
    input_output[idx] = val * scale;
}

// Wrapper functions
torch::Tensor conv_transpose_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(1);  // Note: weight is [in_channels, out_channels, k, k]
    
    int out_height = (in_height - 1) * stride + kernel_size - 2 * padding + output_padding;
    int out_width = (in_width - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::device(input.device()).dtype(input.dtype()));
    
    dim3 threads(32, 4, 4);
    dim3 blocks(
        (out_channels + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        (out_width + threads.z - 1) / threads.z
    );
    
    conv_transpose_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding);
    
    return output;
}

void add_hardtanh_scale_cuda(
    torch::Tensor input_output,
    float add_value,
    float scale) {
    
    int num_elements = input_output.numel();
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    add_hardtanh_scale_kernel<<<num_blocks, block_size>>>(
        input_output.data_ptr<float>(),
        add_value,
        scale,
        num_elements);
}
"""

custom_ops_cpp_source = """
torch::Tensor conv_transpose_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding);
void add_hardtanh_scale_cuda(torch::Tensor input_output, float add_value, float scale);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_ops_cpp_source,
    cuda_sources=custom_ops_source,
    functions=["conv_transpose_mish_cuda", "add_hardtanh_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale
        
        # Initialize weights with correct shape [in_channels, out_channels, k, k]
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Fused conv_transpose + Mish
        x = custom_ops.conv_transpose_mish_cuda(
            x, 
            self.weight, 
            self.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding)
        
        # Fused add + hardtanh + scale
        custom_ops.add_hardtanh_scale_cuda(x, self.add_value, self.scale)
        
        return x