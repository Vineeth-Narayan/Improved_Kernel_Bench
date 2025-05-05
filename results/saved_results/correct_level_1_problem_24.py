# runtime: 0.0397
# basline: 0.0259
# speedup: 0.6523929471032746
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math # Needed for CUDA kernel defines like INFINITY

# CUDA source code for LogSoftmax
log_softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf, logf
#include <limits> // For std::numeric_limits

// Define INFINITY if not defined by CUDA headers (might be needed depending on version/includes)
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif


// Helper device function for block-wide reduction using shared memory
template <typename T>
__device__ T block_reduce_sum(T val, float* sdata) {
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0]; // Result is in the first element
}

// Helper device function for block-wide max reduction using shared memory
template <typename T>
__device__ T block_reduce_max(T val, float* sdata) {
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]); // Use CUDA's max
        }
        __syncthreads();
    }
    return sdata[0]; // Result is in the first element
}


__global__ void log_softmax_kernel(const float* x, float* out, int batch_size, int dim) {
    // Shared memory for reduction operations (max and sum)
    // Size needs to accommodate blockDim.x floats
    extern __shared__ float sdata[];

    int row_idx = blockIdx.x; // Each block handles one row (one batch element)
    int tid = threadIdx.x;    // Thread index within the block
    int block_dim = blockDim.x; // Number of threads in the block

    // Calculate pointers to the start of the current row for input and output
    const float* row_x = x + row_idx * dim;
    float* row_out = out + row_idx * dim;

    // --- Step 1: Find max value in the row (Parallel Reduction) ---
    float thread_max_val = -std::numeric_limits<float>::infinity();
    // Each thread processes multiple elements if dim > block_dim (grid-stride loop)
    for (int j = tid; j < dim; j += block_dim) {
        thread_max_val = max(thread_max_val, row_x[j]); // Use CUDA's max
    }

    // Reduce max values across the block
    float row_max_val = block_reduce_max(thread_max_val, sdata);
    // After reduction, row_max_val holds the maximum value for the current row for all threads in the block
     __syncthreads(); // Ensure all threads have the correct row_max_val before proceeding


    // --- Step 2: Calculate sum of exponentials (shifted by max) ---
    float thread_sum_exp = 0.0f;
    // Each thread processes multiple elements
    for (int j = tid; j < dim; j += block_dim) {
        thread_sum_exp += expf(row_x[j] - row_max_val); // Use expf for float
    }

    // Reduce sum_exp across the block
    float row_sum_exp = block_reduce_sum(thread_sum_exp, sdata);
    // After reduction, row_sum_exp holds the sum for the current row for all threads in the block
     __syncthreads(); // Ensure all threads have the correct row_sum_exp

    // --- Step 3: Calculate log of the sum ---
    // Only one thread needs to do this, but all threads need the result
    // Let thread 0 calculate it and store in shared memory (or just recalculate)
    // For simplicity here, all threads calculate it.
    float log_sum_exp = logf(row_sum_exp); // Use logf for float

    // --- Step 4: Calculate final LogSoftmax values and write to output ---
    // Each thread calculates and writes its portion of the output row
    for (int j = tid; j < dim; j += block_dim) {
        row_out[j] = (row_x[j] - row_max_val) - log_sum_exp;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor x, int64_t dim_arg) {
    // Basic input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D (batch_size, dim)");
    TORCH_CHECK(dim_arg == 1, "Custom LogSoftmax currently only supports dim=1");

    int batch_size = x.size(0);
    int dim_size = x.size(1);

    // Allocate output tensor
    auto out = torch::empty_like(x);

    // Choose CUDA launch parameters
    // Using 512 threads per block is often a good starting point
    const int block_size = 512;
    // One block per row (batch element)
    const int num_blocks = batch_size;
    // Shared memory size: block_size floats for reduction helpers
    const int shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    log_softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim_size
    );

    // Check for kernel launch errors (optional but recommended for debugging)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));


    return out;
}
"""

# C++ source for function declaration (needed by load_inline)
log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor x, int64_t dim_arg);
"""

# Compile the inline CUDA code
log_softmax_module = load_inline(
    name="log_softmax_module",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_cuda_source,
    functions=["log_softmax_cuda"],
    verbose=True, # Set to False for cleaner output once compiled
    extra_cuda_cflags=["-std=c++17"], # Ensure C++17 for std::numeric_limits if needed
    # Add other flags if necessary, e.g., architecture flags like -gencode=arch=compute_75,code=sm_75
)

class ModelNew(nn.Module):
    """
    Simple model that performs a LogSoftmax activation using a custom CUDA kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        # Store the custom function handle
        self.custom_log_softmax = log_softmax_module.log_softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom CUDA LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim) residing on CUDA device.

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # Ensure input is on CUDA before calling the custom kernel
        if not x.is_cuda:
             x = x.cuda() # Move to GPU if not already there

        # Call the custom CUDA function
        return self.custom_log_softmax(x, self.dim)

# Define batch_size and dim consistent with the original model definition
batch_size = 16
dim = 16384

def get_inputs():
    # Ensure inputs are generated on the correct device (CUDA)
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return [] # No special initialization inputs needed