# runtime: 0.0251
# basline: 0.0258
# speedup: 1.0278884462151394
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sigmoid activation
sigmoid_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), 
                                              output.data_ptr<float>(), 
                                              size);
    
    return output;
}
"""

sigmoid_cpp_source = "torch::Tensor sigmoid_cuda(torch::Tensor input);"

# Load the custom CUDA kernel
sigmoid_op = load_inline(
    name="sigmoid_op",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_kernel_source,
    functions=["sigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Sigmoid activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid = sigmoid_op
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return self.sigmoid.sigmoid_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed