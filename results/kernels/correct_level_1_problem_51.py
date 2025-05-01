import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void argmax_kernel(const T* input, int64_t* output, 
                             int dim_size, int stride, int outer_size, int inner_size) {
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    for (int inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
        const T* dim_start = input + outer_idx * dim_size * inner_size + inner_idx;
        T max_val = dim_start[0];
        int max_idx = 0;

        for (int i = 1; i < dim_size; ++i) {
            if (dim_start[i * inner_size] > max_val) {
                max_val = dim_start[i * inner_size];
                max_idx = i;
            }
        }
        
        output[outer_idx * inner_size + inner_idx] = max_idx;
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    // Ensure contiguous memory
    input = input.contiguous();
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    int dim_size = sizes[dim];
    
    // Calculate outer and inner dimensions
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); ++i) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor
    auto options = torch::TensorOptions()
                    .dtype(torch::kLong)
                    .device(input.device());
    auto output = torch::empty({outer_size, inner_size}, options);
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (outer_size + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            dim_size,
            1,  // stride is now handled via inner_size
            outer_size,
            inner_size
        );
    }));
    
    // Reshape output to match input dimensions (minus the reduced dimension)
    std::vector<int64_t> output_shape;
    for (int i = 0; i < sizes.size(); ++i) {
        if (i != dim) output_shape.push_back(sizes[i]);
    }
    
    return output.view(output_shape);
}
"""

argmax_cpp_source = "torch::Tensor argmax_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
argmax_cuda = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_cuda = argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return self.argmax_cuda.argmax_cuda(x, self.dim)

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]