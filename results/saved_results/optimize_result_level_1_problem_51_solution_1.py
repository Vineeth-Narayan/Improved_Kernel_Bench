# Problem Name: 51_Argmax_over_a_dimension
# optimized kernel after 3 iterations
# runtime: 0.0863
# baseline: 0.0396
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the optimized custom CUDA kernel for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void argmax_kernel(const T* __restrict__ input, int64_t* __restrict__ output, 
                             int outer_size, int dim_size, int inner_size) {
    extern __shared__ char shared_mem[];
    T* sdata = (T*)shared_mem;
    int64_t* sidx = (int64_t*)(shared_mem + dim_size * sizeof(T));
    
    const int outer_idx = blockIdx.x;
    const int inner_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const T* current_input = input + outer_idx * dim_size * inner_size + inner_idx;
    int64_t* current_output = output + outer_idx * inner_size + inner_idx;
    
    // Initialize shared memory with coalesced reads
    if (tid < dim_size) {
        sdata[tid] = current_input[tid * inner_size];
        sidx[tid] = tid;
    }
    __syncthreads();
    
    // Optimized parallel reduction with less divergence
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && tid + s < dim_size) {
            if (sdata[tid] < sdata[tid + s] || 
                (sdata[tid] == sdata[tid + s] && sidx[tid] > sidx[tid + s])) {
                sdata[tid] = sdata[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result with single thread per output
    if (tid == 0) {
        *current_output = sidx[0];
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim) {
    // Ensure dim is positive
    dim = dim < 0 ? dim + input.dim() : dim;
    
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    
    auto output = torch::empty(sizes, torch::dtype(torch::kLong).device(input.device()));
    
    // Calculate dimensions for kernel launch
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= input.size(i);
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < input.dim(); ++i) {
        inner_size *= input.size(i);
    }
    
    // Optimized kernel launch configuration
    const int max_threads = 256;  // Better for T4's 64KB shared memory
    int threads = std::min(max_threads, static_cast<int>((dim_size + 31) / 32 * 32));
    dim3 blocks(outer_size, inner_size);
    
    size_t shared_mem_size = dim_size * (sizeof(float) + sizeof(int64_t));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", [&] {
        argmax_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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

argmax_cpp_source = "torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim);"

# Compile the inline CUDA code for argmax
argmax_cuda = load_inline(
    name="argmax_cuda",
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
        self.argmax_cuda = argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return self.argmax_cuda.argmax_cuda(x, self.dim)