# runtime: 0.0408
# basline: 0.072
# speedup: 1.764705882352941
import torch
import torch.nn as nn
import math  # Added this import
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose1d
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_length) {
    
    // Calculate output position
    const int out_ch = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_pos = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_ch < out_channels && out_pos < output_length && batch < batch_size) {
        float value = 0.0f;
        
        // Loop over input channels
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            // Loop over kernel positions
            for (int k = 0; k < kernel_size; ++k) {
                // Calculate input position (reverse of regular conv)
                int input_pos = (out_pos - k * dilation + padding) / stride;
                
                // Check boundaries
                if (input_pos >= 0 && input_pos < input_length && 
                    (out_pos - k * dilation + padding) % stride == 0) {
                    
                    // Get weight and input values
                    float w = weight[in_ch * out_channels * kernel_size + out_ch * kernel_size + k];
                    float x = input[batch * in_channels * input_length + in_ch * input_length + input_pos];
                    value += w * x;
                }
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            value += bias[out_ch];
        }
        
        // Store result
        output[batch * out_channels * output_length + out_ch * output_length + out_pos] = value;
    }
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
    
    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    
    // Get weight dimensions
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    // Calculate output length
    int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());
    
    // Set up CUDA blocks and threads
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (out_channels + threads.x - 1) / threads.x,
        (output_length + threads.y - 1) / threads.y,
        (batch_size + threads.z - 1) / threads.z
    );
    
    // Get pointers
    float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    // Launch kernel
    conv_transpose1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length
    );
    
    return output;
}
"""

conv_transpose1d_cpp_source = """
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation);
"""

# Compile the inline CUDA code
conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weight parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose1d.conv_transpose1d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.Tensor(),  # Empty tensor for nullptr
            self.stride,
            self.padding,
            self.dilation
        )