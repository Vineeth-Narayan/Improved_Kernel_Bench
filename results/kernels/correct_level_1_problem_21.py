import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sigmoid activation
sigmoid_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    sigmoid_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

sigmoid_cpp_source = "torch::Tensor sigmoid_cuda(torch::Tensor input);"

# Compile the inline CUDA code
sigmoid_module = load_inline(
    name="sigmoid_module",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_kernel_source,
    functions=["sigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for Sigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Sigmoid activation using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return sigmoid_module.sigmoid_cuda(x.cuda())

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed