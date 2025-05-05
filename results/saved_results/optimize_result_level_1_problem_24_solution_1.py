# Problem Name: 24_LogSoftmax
# optimized kernel after 5 iterations
# runtime: 0.034
# baseline: 0.0259
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math # Needed for CUDA kernel defines like INFINITY
import os # For setting CUDA architecture flags if needed

# Set TORCH_CUDA_ARCH_LIST if it's not set, to avoid potential warnings/errors during compilation
# Example: Set for Turing (7.5) and Ampere (8.0, 8.6). Adjust based on your target GPU.
# if "TORCH_CUDA_ARCH_LIST" not in os.environ:
#     os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6"

# CUDA source code for LogSoftmax with float4 vectorization
log_softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf, logf, fmaxf
#include <limits> // For std::numeric_limits
#include <vector_types.h> // For float4

// Define INFINITY if not defined by CUDA headers (less common now, but safe)
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NEG_INFINITY
#define NEG_INFINITY (__int_as_float(0xff800000))
#endif


// Helper device function for block-wide reduction using shared memory
// This function operates on scalar float values. Each thread contributes one value.
// Uses warp shuffle for the final reduction steps.
__device__ float block_reduce_sum(float val, float* sdata) {
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    sdata[tid] = val;
    __syncthreads();

    // --- Inter-warp reduction using shared memory ---
    // Reduce sums from each warp into the first warp's portion of shared memory
    if (block_dim >= 64) { // Only needed if more than one warp
        // Reduce across warps. Each thread reduces values 's' positions apart.
        for (unsigned int s = block_dim / 2; s >= 32; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads(); // Sync after each reduction level
        }
    }

    // --- Intra-warp reduction using warp shuffle (for the first warp) ---
    // Only threads in the first warp participate in the final shuffle reduction.
    if (warp_id == 0) {
        // volatile qualifier prevents compiler optimizations that might interfere
        volatile float* warp_sdata = sdata;
        float warp_sum = warp_sdata[tid]; // Load my value (tid == lane_id here)

        // Use __shfl_down_sync for reduction within the warp
        // This sums values downwards across lanes.
        // Mask 0xffffffff means all threads in the warp participate.
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        // The final sum resides in lane 0 of the warp.
        // Write the warp's sum back to shared memory (optional, but good practice)
        if (lane_id == 0) {
            sdata[0] = warp_sum;
        }
    }

    // Ensure the final result in sdata[0] is visible to all threads
    __syncthreads();
    return sdata[0]; // Result is in the first element
}

// Helper device function for block-wide max reduction using shared memory
// Uses warp shuffle for the final reduction steps.
__device__ float block_reduce_max(float val, float* sdata) {
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    sdata[tid] = val;
    __syncthreads();

    // --- Inter-warp reduction using shared memory ---
    if (block_dim >= 64) {
        for (unsigned int s = block_dim / 2; s >= 32; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]); // Use fmaxf for float comparison
            }
            __syncthreads();
        }
    }

    // --- Intra-warp reduction using warp shuffle (for the first warp) ---
    if (warp_id == 0) {
        volatile float* warp_sdata = sdata;
        float warp_max = warp_sdata[tid]; // Load my value

        // Use __shfl_down_sync for reduction within the warp
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
        }
        // The final max resides in lane 0 of the warp.
        if (lane_id == 0) {
            sdata[0] = warp_max;
        }
    }

    __syncthreads(); // Ensure all writes are visible
    return sdata[0]; // Result is in the first element
}


__global__ void log_softmax_vectorized_kernel(const float* __restrict__ x, float* __restrict__ out, int batch_size, int dim) {
    // Shared memory for reduction operations (max and sum)
    // Allocated dynamically based on launch configuration
    extern __shared__ float sdata[];

    // This kernel assumes dim is divisible by 4. Check is done in host code.
    int dim4 = dim / 4; // Dimension in terms of float4 vectors

    int row_idx = blockIdx.x; // Each block handles one row (one batch element)
    int tid = threadIdx.x;    // Thread index within the block
    int block_dim = blockDim.x; // Number of threads in the block

    // Bounds check: only process valid rows
    if (row_idx >= batch_size) {
        return;
    }

    // Calculate pointers to the start of the current row for input and output
    // Use float4 pointers for vectorized access
    const float4* row_x4 = reinterpret_cast<const float4*>(x + row_idx * dim);
    float4*       row_out4 = reinterpret_cast<float4*>(out + row_idx * dim);

    // --- Step 1: Find max value in the row (Parallel Reduction with float4 loads) ---
    // Initialize thread max to negative infinity
    float thread_max_val = NEG_INFINITY; // Use defined NEG_INFINITY

    // Grid-stride loop: Each thread processes multiple float4 elements if dim4 > block_dim
    for (int j = tid; j < dim4; j += block_dim) {
        float4 current_x4 = row_x4[j];
        // Find max within the float4 vector using fmaxf
        float m1 = fmaxf(current_x4.x, current_x4.y);
        float m2 = fmaxf(current_x4.z, current_x4.w);
        float vec_max = fmaxf(m1, m2);
        // Update thread's max value
        thread_max_val = fmaxf(thread_max_val, vec_max);
    }

    // Reduce max values across the block using the helper function
    // Pass the shared memory pointer `sdata`
    float row_max_val = block_reduce_max(thread_max_val, sdata);
    // After reduction, row_max_val holds the maximum value for the current row.
    // Sync threads to ensure all threads have the correct row_max_val before proceeding.
    // Note: block_reduce_max already includes necessary __syncthreads() internally.


    // --- Step 2: Calculate sum of exponentials (shifted by max) ---
    float thread_sum_exp = 0.0f;
    // Grid-stride loop again
    for (int j = tid; j < dim4; j += block_dim) {
        float4 current_x4 = row_x4[j];
        // Calculate exponentials for the vector, shifted by row_max_val
        float exp_x = expf(current_x4.x - row_max_val);
        float exp_y = expf(current_x4.y - row_max_val);
        float exp_z = expf(current_x4.z - row_max_val);
        float exp_w = expf(current_x4.w - row_max_val);
        // Sum the exponentials within the vector
        float vec_sum = exp_x + exp_y + exp_z + exp_w;
        // Accumulate sum for the thread
        thread_sum_exp += vec_sum;
    }

    // Reduce sum_exp across the block
    // Pass the shared memory pointer `sdata`
    float row_sum_exp = block_reduce_sum(thread_sum_exp, sdata);
    // Sync threads to ensure all threads have the correct row_sum_exp.
    // Note: block_reduce_sum already includes necessary __syncthreads() internally.

    // --- Step 3: Calculate log of the sum ---
    // Only one thread needs to calculate this, but all threads need the result.
    // Let thread 0 calculate and store it in shared memory (or just use the value directly).
    // Using the value directly is fine as row_sum_exp is consistent across threads after reduction.
    float log_sum_exp = logf(row_sum_exp);

    // --- Step 4: Calculate final LogSoftmax values and write to output using float4 ---
    // Grid-stride loop for writing output
    for (int j = tid; j < dim4; j += block_dim) {
        float4 current_x4 = row_x4[j]; // Read input vector again
        float4 out4;
        // Calculate LogSoftmax for each element in the vector
        out4.x = (current_x4.x - row_max_val) - log_sum_exp;
        out4.y = (current_x4.y - row_max_val) - log_sum_exp;
        out4.z = (current_x4.z - row_max_val) - log_sum_exp;
        out4.w = (current_x4.w - row_max_val) - log_sum_exp;
        // Write the output vector
        row_out4[j] = out4;
    }
}

// C++ interface function (called from Python)
torch::Tensor log_softmax_cuda(torch::Tensor x, int64_t dim_arg) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D (batch_size, dim)");
    TORCH_CHECK(dim_arg == 1, "Custom LogSoftmax currently only supports dim=1");
    TORCH_CHECK(x.size(1) > 0, "Input dimension (dim=1) must be greater than 0");
    TORCH_CHECK(x.size(1) % 4 == 0, "Input dimension (dim=1) must be divisible by 4 for vectorized kernel");
    // Ensure tensor strides allow for coalesced float4 access (last dimension should be contiguous)
    TORCH_CHECK(x.is_contiguous() || x.stride(1) == 1, "Input tensor must be contiguous or have stride 1 along the reduction dimension (dim=1)");


    int batch_size = x.size(0);
    int dim_size = x.size(1);

    // Ensure input is contiguous if necessary (safer)
    // The kernel relies on contiguous access within a row.
    auto x_contig = x.contiguous();


    // Allocate output tensor
    auto out = torch::empty_like(x_contig);

    // Choose CUDA launch parameters
    // Using 512 threads per block is often a good starting point for reduction-heavy kernels.
    // Could tune this (e.g., 256, 1024) based on specific GPU and problem size.
    // Max threads per block is typically 1024.
    const int block_size = 512;
    // One block per row (batch element)
    const int num_blocks = batch_size;
    // Shared memory size: block_size floats for reduction helpers
    // Needs to be large enough for block_reduce_sum/max.
    const int shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    log_softmax_vectorized_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x_contig.data_ptr<float>(), // Use contiguous input
        out.data_ptr<float>(),
        batch_size,
        dim_size
    );

    // Check for kernel launch errors immediately after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));
    // Optional: Synchronize device to check for runtime errors within the kernel
    // It's good practice during development/debugging. Can be removed for performance.
    // cudaDeviceSynchronize();
    // err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel runtime error: ", cudaGetErrorString(err));


    return out;
}
"""

# C++ source for function declaration (needed by load_inline)
log_softmax_cpp_source = """
#include <torch/extension.h>
// Declaration must match the definition in the CUDA source
torch::Tensor log_softmax_cuda(torch::Tensor x, int64_t dim_arg);
"""

# Compile the inline CUDA code
# It's good practice to set verbose=False once compilation is successful
try:
    log_softmax_module = load_inline(
        name="log_softmax_module_optimized_v2", # Changed name again to ensure clean build
        cpp_sources=log_softmax_cpp_source,
        cuda_sources=log_softmax_cuda_source,
        functions=["log_softmax_cuda"],
        verbose=True, # Set to True for debugging compilation
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3", # Enable optimizations
            # Add architecture flag for common architectures if needed, or rely on TORCH_CUDA_ARCH_LIST
            # "-gencode=arch=compute_75,code=sm_75", # Example: Turing
            # "-gencode=arch=compute_86,code=sm_86"  # Example: Ampere
            ]
    )
    print("CUDA extension compiled successfully.")
except Exception as e:
    print(f"Failed to compile CUDA extension: {e}")
    # Raise the exception to halt execution if compilation fails
    raise e

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for LogSoftmax
    with float4 vectorization and improved reduction.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        # Basic validation matching kernel requirements
        if dim != 1:
             raise ValueError("Custom LogSoftmax currently only supports dim=1")
        self.dim = dim
        # Store the custom function handle
        # Ensure the module was loaded correctly before accessing the function
        if log_softmax_module is None:
             raise RuntimeError("CUDA extension module failed to load.")
        self.custom_log_softmax = log_softmax_module.log_softmax_cuda

    # FIX: Corrected the return type annotation from torch::Tensor to torch.Tensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom CUDA LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim) residing on CUDA device.
                              dim must be divisible by 4.

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # Ensure input is on CUDA before calling the custom kernel
        if not x.is_cuda:
             # Move to GPU if not already there. Ensure it's the default CUDA device
             # or match the device of the model parameters if applicable.
             x = x.cuda()

        # Input validation specific to this kernel's requirements (can be done here or in C++)
        # C++ checks are generally preferred as they catch issues closer to the kernel call.
        # The C++ checks are already implemented in log_softmax_cuda.
        # if x.dim() != 2:
        #     raise ValueError(f"Input tensor must be 2D (batch_size, dim), got {x.dim()}D")
        # if x.size(1) % 4 != 0:
        #      raise ValueError(f"Input dimension (dim=1) must be divisible by 4 for vectorized kernel, got {x.size(1)}")
        # if x.stride(1) != 1 and not x.is_contiguous(): # Check contiguity or stride
        #      raise ValueError("Input tensor must be contiguous or have stride 1 along the reduction dimension (dim=1)")


        # Call the custom CUDA function
        # The C++ function handles contiguity check/enforcement now.
        return self.custom_log_softmax(x, self.dim)

# Define batch_size and dim consistent with the original model definition
batch_size = 16
dim = 16384 # Divisible by 4, suitable for float4

def get_inputs():
    # Ensure inputs are generated on the correct device (CUDA) and are float32
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cuda')
    # No need to explicitly call .contiguous() here, as the C++ wrapper handles it.
    # However, ensuring inputs are contiguous beforehand can sometimes avoid the overhead
    # of the copy inside the C++ wrapper if the tensor wasn't already contiguous.
    # x = x.contiguous()
    return [x]

def get_init_inputs():
    return [] # No special initialization inputs needed

# Example usage (optional, for testing)
if __name__ == '__main__':
    print("Testing custom LogSoftmax kernel...")
    model_orig = Model(dim=1).cuda()
    model_custom = ModelNew(dim=1).cuda() # ModelNew expects CUDA inputs

    # Get inputs (already on CUDA)
    inputs = get_inputs()
    x_input = inputs[0]

    # Ensure models are in eval mode if they had dropout/batchnorm layers
    model_orig.eval()
    model_custom.eval()

    # Run original PyTorch version
    with torch.no_grad():
        output_orig = model_orig(x_input)

    # Run custom CUDA version
    with torch.no_grad():
        output_custom = model_custom(x_input)

    # Compare results
    print("Comparing outputs...")
    print("Max difference:", torch.max(torch.abs(output_orig - output_custom)))
    print("Mean difference:", torch.mean(torch.abs(output_orig - output_custom)))

    # Check if outputs are close enough
    are_close = torch.allclose(output_orig, output_custom, atol=1e-5, rtol=1e-4) # Adjust tolerance as needed
    print(f"Outputs are close: {are_close}")

    if not are_close:
        print("Warning: Outputs differ significantly!")

    # Simple benchmark (optional)
    import time
    n_runs = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_runs):
        model_orig(x_input)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Original PyTorch LogSoftmax time: {(end_time - start_time) / n_runs * 1000:.4f} ms")

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_runs):
        model_custom(x_input)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Custom CUDA LogSoftmax time: {(end_time - start_time) / n_runs * 1000:.4f} ms")