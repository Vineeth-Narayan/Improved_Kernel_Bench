# runtime: 0.172
# basline: 0.0396
# speedup: 0.23023255813953492
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void argmax_kernel(const T* input, int64_t* output, 
                             int outer_size, int dim_size, int inner_size) {
    extern __shared__ char shared_mem[];
    T* sdata = (T*)shared_mem;
    int64_t* sidx = (int64_t*)(shared_mem + dim_size * sizeof(T));
    
    int outer_idx = blockIdx.x / inner_size;
    int inner_idx = blockIdx.x % inner_size;
    int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Initialize shared memory
    if (threadIdx.x < dim_size) {
        sdata[threadIdx.x] = input[input_offset + threadIdx.x * inner_size];
        sidx[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();
    
    // Parallel reduction to find max value and index
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < dim_size) {
            if (sdata[threadIdx.x] < sdata[threadIdx.x + s] || 
                (sdata[threadIdx.x] == sdata[threadIdx.x + s] && sidx[threadIdx.x] > sidx[threadIdx.x + s])) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sidx[0];
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    // Ensure input is contiguous
    input = input.contiguous();
    
    // Get dimensions
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
    auto output_sizes = sizes.vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, torch::dtype(torch::kLong).device(input.device()));
    
    // Determine kernel configuration
    int block_size = 256;
    while (block_size > dim_size * 2) {
        block_size >>= 1;
    }
    
    dim3 grid(outer_size * inner_size);
    dim3 block(block_size);
    
    // Calculate shared memory size
    size_t shared_mem_size = dim_size * (sizeof(float) + sizeof(int64_t));
    
    // Launch kernel based on input type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", [&] {
        argmax_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    });
    
    return output;
}
"""

argmax_cpp_source = "torch::Tensor argmax_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code for argmax
argmax = load_inline(
    name="argmax",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
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
        self.argmax = argmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return self.argmax.argmax_cuda(x, self.dim)