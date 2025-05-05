# runtime: 0.0137
# basline: 0.0259
# speedup: 1.8905109489051093
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math # Needed for CUDA kernel defines like INFINITY
import os # For setting CUDA architecture flags if needed

# Set TORCH_CUDA_ARCH_LIST for T4 (compute capability 7.5) if not set
# This helps ensure the kernel is compiled optimally for the target GPU.
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

# CUDA source code for LogSoftmax with float4 vectorization and optimized reduction
log_softmax_cuda_source_optimized = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf, logf, fmaxf
#include <limits> // For std::numeric_limits
#include <vector_types.h> // For float4

// Define floating point constants
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NEG_INFINITY
#define NEG_INFINITY (__int_as_float(0xff800000))
#endif

// --- Optimized Block-wise Reduction Kernels ---
// These versions prioritize warp shuffle operations for intra-warp reduction,
// potentially reducing shared memory latency compared to full shared memory reductions.

// Optimized block-wide max reduction using warp shuffles and shared memory for inter-warp communication.
// Assumes blockDim.x is a multiple of 32.
__device__ float block_reduce_max_optimized(float val, float* sdata) {
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    int warp_id = tid / 32; // Warp index within the block
    int lane_id = tid % 32; // Thread index within the warp
    // Calculate the number of warps participating. ceil(block_dim / 32.0)
    // Simplified assuming block_dim is power of 2 and multiple of 32
    int num_warps = block_dim / 32;

    // 1. Intra-warp reduction using shuffle down instructions
    // Each thread finds the maximum value within its warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        // 0xffffffff mask: all threads in the warp participate
    }
    // After this loop, lane 0 of each warp holds the maximum value for that warp.

    // 2. Store warp maximums in shared memory
    // Only lane 0 of each warp writes its result.
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }

    // Synchronize threads within the block to ensure all warp maximums are written to sdata
    __syncthreads();

    // 3. Inter-warp reduction (performed by the first warp)
    // Load warp maximums from shared memory into the first warp's threads.
    // Only threads with tid < num_warps load valid data.
    val = (tid < num_warps) ? sdata[tid] : NEG_INFINITY;

    // Synchronize before the first warp modifies sdata again (optional but safe)
    // __syncthreads(); // Not strictly necessary if only first warp writes below

    // Reduce the warp maximums within the first warp using shuffles.
    if (warp_id == 0) { // Only threads in the first warp participate
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            // Reduce max values potentially held by the first 'num_warps' threads
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        // The final maximum value for the block resides in lane 0 of the first warp.
        // Write the final result back to shared memory (position 0) for all threads to access.
        if (lane_id == 0) {
            sdata[0] = val;
        }
    }

    // Synchronize threads to ensure the final result in sdata[0] is visible to all threads
    __syncthreads();

    // Return the block-wide maximum value
    return sdata[0];
}


// Optimized block-wide sum reduction using warp shuffles and shared memory for inter-warp communication.
// Assumes blockDim.x is a multiple of 32.
__device__ float block_reduce_sum_optimized(float val, float* sdata) {
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = block_dim / 32;

    // 1. Intra-warp reduction using shuffle down
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // Lane 0 of each warp now holds the sum for that warp.

    // 2. Store warp sums in shared memory
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }

    // Synchronize to ensure all warp sums are written
    __syncthreads();

    // 3. Inter-warp reduction (performed by the first warp)
    // Load warp sums into the first warp's threads.
    val = (tid < num_warps) ? sdata[tid] : 0.0f;

    // Synchronize before the first warp modifies sdata again (optional but safe)
    // __syncthreads();

    // Reduce the warp sums within the first warp using shuffles.
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        // Final sum in lane 0 of the first warp.
        if (lane_id == 0) {
            sdata[0] = val; // Write final sum
        }
    }

    // Synchronize to make the final sum visible to all threads
    __syncthreads();

    return sdata[0];
}


// Kernel using float4 vectorization and optimized reductions
__global__ void log_softmax_vectorized_kernel_optimized(const float* __restrict__ x, float* __restrict__ out, int batch_size, int dim) {
    // Shared memory for reduction operations. Size should be sufficient for the reduction strategy.
    // Max needed is blockDim.x floats for the older strategy, or num_warps floats for the optimized one.
    // Allocating blockDim.x floats is safe and covers both.
    extern __shared__ float sdata[];

    // Assume dim is divisible by 4 (checked in host code)
    const int dim4 = dim / 4; // Dimension in terms of float4 vectors

    const int row_idx = blockIdx.x; // Each block handles one row
    const int tid = threadIdx.x;    // Thread index within the block
    const int block_dim = blockDim.x; // Number of threads per block

    // Bounds check for rows
    if (row_idx >= batch_size) {
        return;
    }

    // Pointers for vectorized access within the current row
    const float4* row_x4 = reinterpret_cast<const float4*>(x + row_idx * dim);
    float4*       row_out4 = reinterpret_cast<float4*>(out + row_idx * dim);

    // --- Step 1: Find max value in the row (Optimized Reduction) ---
    float thread_max_val = NEG_INFINITY;
    // Grid-stride loop for elements within the row
    for (int j = tid; j < dim4; j += block_dim) {
        float4 current_x4 = row_x4[j];
        // Max within float4
        float m1 = fmaxf(current_x4.x, current_x4.y);
        float m2 = fmaxf(current_x4.z, current_x4.w);
        float vec_max = fmaxf(m1, m2);
        // Update thread max
        thread_max_val = fmaxf(thread_max_val, vec_max);
    }

    // Reduce max values across the block using the optimized helper
    const float row_max_val = block_reduce_max_optimized(thread_max_val, sdata);
    // Synchronization is handled within block_reduce_max_optimized

    // --- Step 2: Calculate sum of exponentials (shifted by max) ---
    float thread_sum_exp = 0.0f;
    // Grid-stride loop again
    for (int j = tid; j < dim4; j += block_dim) {
        float4 current_x4 = row_x4[j];
        // Shifted exponentials for float4
        float exp_x = expf(current_x4.x - row_max_val);
        float exp_y = expf(current_x4.y - row_max_val);
        float exp_z = expf(current_x4.z - row_max_val);
        float exp_w = expf(current_x4.w - row_max_val);
        // Accumulate sum for the thread
        thread_sum_exp += (exp_x + exp_y + exp_z + exp_w);
    }

    // Reduce sum_exp across the block using the optimized helper
    const float row_sum_exp = block_reduce_sum_optimized(thread_sum_exp, sdata);
    // Synchronization is handled within block_reduce_sum_optimized

    // --- Step 3: Calculate log of the sum ---
    // This value is consistent across all threads in the block after reduction.
    const float log_sum_exp = logf(row_sum_exp);

    // --- Step 4: Calculate final LogSoftmax values and write output ---
    // Grid-stride loop for writing output
    for (int j = tid; j < dim4; j += block_dim) {
        const float4 current_x4 = row_x4[j]; // Read input vector again
        float4 out4;
        // Calculate log_softmax = (x - max) - log(sum(exp(x - max)))
        out4.x = (current_x4.x - row_max_val) - log_sum_exp;
        out4.y = (current_x4.y - row_max_val) - log_sum_exp;
        out4.z = (current_x4.z - row_max_val) - log_sum_exp;
        out4.w = (current_x4.w - row_max_val) - log_sum_exp;
        // Write the output vector
        row_out4[j] = out4;
    }
}

// C++ interface function (called from Python)
torch::Tensor log_softmax_cuda_optimized(torch::Tensor x, int64_t dim_arg) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D (batch_size, dim)");
    TORCH_CHECK(dim_arg == 1, "Custom LogSoftmax currently only supports dim=1");
    const int batch_size = x.size(0);
    const int dim_size = x.size(1);
    TORCH_CHECK(dim_size > 0, "Input dimension (dim=1) must be greater than 0");
    // Kernel requires dim divisible by 4 for float4 vectorization
    TORCH_CHECK(dim_size % 4 == 0, "Input dimension (dim=1) must be divisible by 4 for vectorized kernel");
    // Ensure contiguity for coalesced float4 access along the reduction dimension.
    // The kernel reads row_x4[j] where j increments linearly.
    TORCH_CHECK(x.stride(0) >= dim_size && x.stride(1) == 1, "Input tensor must be contiguous or effectively row-contiguous (stride(1)==1)");

    // Ensure input is contiguous (or handle non-contiguous layout if necessary)
    // Making it contiguous simplifies kernel logic significantly.
    auto x_contig = x.contiguous();

    // Allocate output tensor
    auto out = torch::empty_like(x_contig);

    // --- CUDA Kernel Launch Configuration ---
    // Block size: Powers of 2, typically 128, 256, 512. Must be multiple of 32 for warp shuffles.
    // 512 was used previously and is often a good balance for reduction kernels.
    // Tuning this (e.g., to 256 or 1024) might yield improvements based on specific GPU occupancy/latency.
    const int block_size = 512;
    // Grid size: One block per row (batch element).
    const int num_blocks = batch_size;
    // Shared memory size: Sufficient for the reduction helpers.
    // block_reduce_*_optimized needs max(num_warps * sizeof(float)).
    // Allocating block_size * sizeof(float) is safe and covers various strategies.
    const int shared_mem_size = block_size * sizeof(float);

    // Check if requested shared memory exceeds device limits (optional sanity check)
    // int dev; cudaGetDevice(&dev);
    // cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    // TORCH_CHECK(shared_mem_size <= prop.sharedMemPerBlock, "Requested shared memory exceeds device limit per block.");

    // Launch the optimized kernel
    log_softmax_vectorized_kernel_optimized<<<num_blocks, block_size, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim_size
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error in log_softmax_optimized: ", cudaGetErrorString(err));
    // Optional: Synchronize for debugging kernel runtime errors (comment out for performance)
    // cudaDeviceSynchronize();
    // err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel runtime error in log_softmax_optimized: ", cudaGetErrorString(err));

    return out;
}
"""

# C++ source for function declaration (needed by load_inline)
log_softmax_cpp_source_optimized = """
#include <torch/extension.h>
// Declaration must match the definition in the CUDA source
torch::Tensor log_softmax_cuda_optimized(torch::Tensor x, int64_t dim_arg);
"""

# --- Compilation ---
# Unique name to avoid conflicts if run multiple times or with other extensions
module_name = "log_softmax_module_optimized_final"
# Remove previously compiled module files if they exist to ensure a clean build
build_dir = os.path.join(os.getcwd(), "build", module_name) # Example build directory structure
if os.path.exists(build_dir):
    import shutil
    # print(f"Removing existing build directory: {build_dir}")
    # try:
    #     shutil.rmtree(build_dir)
    # except OSError as e:
    #     print(f"Error removing directory {build_dir}: {e}")
    pass # Avoid deleting if not necessary or if permissions are an issue


# Compile the inline CUDA code
# Use verbose=True for debugging compilation issues
try:
    log_softmax_module_optimized = load_inline(
        name=module_name,
        cpp_sources=log_softmax_cpp_source_optimized,
        cuda_sources=log_softmax_cuda_source_optimized,
        functions=["log_softmax_cuda_optimized"],
        verbose=False, # Set to True for compile details
        extra_cuda_cflags=[
            "-std=c++17",       # Use C++17 standard
            "-O3",              # Optimization level 3
            "-U__CUDA_NO_HALF_OPERATORS__", # Define if using half-precision types explicitly
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--use_fast_math", # Enables faster, less precise math functions (e.g., __expf). Use with caution if precision is critical.
            # Target specific architecture (T4 is sm_75)
            "-gencode=arch=compute_75,code=sm_75"
            ]
    )
    # print("Optimized CUDA extension compiled successfully.")
except Exception as e:
    print(f"Failed to compile optimized CUDA extension: {e}")
    raise e

# --- Optimized PyTorch Module ---

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for LogSoftmax
    with float4 vectorization and improved shuffle-based reduction.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        # Basic validation matching kernel requirements
        if dim != 1:
             raise ValueError("Custom LogSoftmax currently only supports dim=1")
        self.dim = dim
        # Store the custom function handle
        if log_softmax_module_optimized is None:
             raise RuntimeError("Optimized CUDA extension module failed to load.")
        self.custom_log_softmax = log_softmax_module_optimized.log_softmax_cuda_optimized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the optimized custom CUDA LogSoftmax activation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim) residing on CUDA device.
                              dim must be divisible by 4.

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # Ensure input is on CUDA device
        if not x.is_cuda:
             # Move to the default CUDA device. Ensure this matches model placement if necessary.
             x = x.cuda()

        # Input validation is primarily handled within the C++ wrapper for robustness.
        # Requirements: 2D tensor, dim=1, dim_size > 0, dim_size % 4 == 0, float32, CUDA tensor.
        # The C++ wrapper also handles the contiguity requirement.

        # Call the optimized custom CUDA function
        return self.custom_log_softmax(x, self.dim)

# --- Input Generation Functions ---
# These remain the same as the problem description requires them, used for benchmarking/testing.

batch_size = 16
dim = 16384 # Must be divisible by 4 for the current kernel

def get_inputs():
    """Generates input tensor on CUDA device."""
    # Ensure inputs are generated on CUDA and are float32
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cuda')
    # The C++ wrapper handles contiguity if needed, but generating contiguous
    # inputs avoids potential overhead if the tensor wasn't already contiguous.
    # return [x.contiguous()] # Explicitly make contiguous
    return [x] # Rely on wrapper's contiguous check/call

def get_init_inputs():
    """Returns inputs needed potentially for model initialization (none needed here)."""
    return []