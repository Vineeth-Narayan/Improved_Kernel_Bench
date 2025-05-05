# runtime: 0.0774
# basline: 0.0259
# speedup: 0.33462532299741604
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size,
    const int inner_size,
    const int outer_size) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_max = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* shared_sum = shared_max + blockDim.x;
    
    const int outer_idx = blockIdx.x;
    const int inner_idx = threadIdx.x;
    
    // Each block processes one outer dimension (batch)
    const scalar_t* input_row = input + outer_idx * dim_size;
    scalar_t* output_row = output + outer_idx * dim_size;
    
    // Step 1: Find max value for numerical stability
    scalar_t thread_max = -INFINITY;
    for (int i = inner_idx; i < dim_size; i += blockDim.x) {
        thread_max = max(thread_max, input_row[i]);
    }
    
    shared_max[inner_idx] = thread_max;
    __syncthreads();
    
    // Reduction to find max in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (inner_idx < stride) {
            shared_max[inner_idx] = max(shared_max[inner_idx], shared_max[inner_idx + stride]);
        }
        __syncthreads();
    }
    
    scalar_t row_max = shared_max[0];
    __syncthreads();
    
    // Step 2: Compute exp(x_i - max) and sum
    scalar_t thread_sum = 0;
    for (int i = inner_idx; i < dim_size; i += blockDim.x) {
        thread_sum += exp(input_row[i] - row_max);
    }
    
    shared_sum[inner_idx] = thread_sum;
    __syncthreads();
    
    // Reduction to find sum in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (inner_idx < stride) {
            shared_sum[inner_idx] += shared_sum[inner_idx + stride];
        }
        __syncthreads();
    }
    
    scalar_t row_sum = shared_sum[0];
    scalar_t log_sum = log(row_sum);
    
    // Step 3: Compute final log_softmax: (x_i - max) - log(sum(exp(x_i - max)))
    for (int i = inner_idx; i < dim_size; i += blockDim.x) {
        output_row[i] = (input_row[i] - row_max) - log_sum;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int64_t dim) {
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    
    int dim_size = input_contig.size(dim);
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= input_contig.size(i);
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < input_contig.dim(); ++i) {
        inner_size *= input_contig.size(i);
    }
    
    // Use 256 threads per block (good balance for most GPUs)
    const int threads = 256;
    const int blocks = outer_size * inner_size;
    
    // Shared memory size: 2 * threads * sizeof(float)
    size_t shared_mem_size = 2 * threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input_contig.scalar_type(), "log_softmax_cuda", ([&] {
        log_softmax_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            outer_size);
    }));
    
    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
log_softmax_ext = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with custom CUDA LogSoftmax implementation.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return log_softmax_ext.log_softmax_cuda(x, self.dim)