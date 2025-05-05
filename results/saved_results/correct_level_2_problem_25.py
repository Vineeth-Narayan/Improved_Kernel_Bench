# runtime: 1.81
# basline: 0.254
# speedup: 0.14033149171270717
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel that fuses conv, min reduction, and double tanh
fused_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

template<typename scalar_t>
__global__ void fused_conv_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Each thread handles one output spatial position for one batch
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || h >= output_h || w >= output_w) return;
    
    // For each output channel, compute convolution and find min
    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    
    for (int oc = 0; oc < out_channels; ++oc) {
        scalar_t val = 0;
        
        // Convolution operation
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int ih = h * stride - padding + kh;
                    const int iw = w * stride - padding + kw;
                    
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        const int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                        const int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        if (bias != nullptr) {
            val += bias[oc];
        }
        
        // Track min across channels
        if (val < min_val) {
            min_val = val;
        }
    }
    
    // Apply double tanh
    scalar_t result = tanh(tanh(min_val));
    
    // Write output (batch_size x 1 x output_h x output_w)
    const int output_idx = (b * output_h + h) * output_w + w;
    output[output_idx] = result;
}

torch::Tensor fused_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, 1, output_h, output_w}, input.options());
    
    // Configure thread blocks
    const int threads_x = 4;
    const int threads_y = 16;
    const int threads_z = 16;
    
    dim3 threads(threads_x, threads_y, threads_z);
    dim3 blocks(
        (batch_size + threads_x - 1) / threads_x,
        (output_h + threads_y - 1) / threads_y,
        (output_w + threads_z - 1) / threads_z
    );
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_cuda", ([&] {
        fused_conv_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            stride,
            padding);
    }));
    
    return output;
}
"""

fused_conv_cpp_source = """
torch::Tensor fused_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding);
"""

# Compile the inline CUDA code
fused_conv = load_inline(
    name="fused_conv",
    cpp_sources=fused_conv_cpp_source,
    cuda_sources=fused_conv_source,
    functions=["fused_conv_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that uses fused CUDA kernel for conv + min + tanh + tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 0  # Changed from kernel_size//2 to match original behavior
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_conv = fused_conv

    def forward(self, x):
        return self.fused_conv.fused_conv_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding
        )