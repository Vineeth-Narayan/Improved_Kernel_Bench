import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void log_softmax_kernel(const float* input, float* output, int dim_size, int batch_size, int dim) {
    // Each block handles one sample in the batch
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_mem[];
    float* max_val = shared_mem;
    float* sum_exp = shared_mem + 1;
    
    const float* batch_input = input + batch_idx * dim_size;
    float* batch_output = output + batch_idx * dim_size;
    
    // Step 1: Find max value in the dimension (for numerical stability)
    max_val[0] = -INFINITY;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = batch_input[i];
        if (val > max_val[0]) {
            atomicMax((int*)max_val, __float_as_int(val));
        }
    }
    __syncthreads();
    
    // Step 2: Compute sum of exp(x_i - max_val)
    sum_exp[0] = 0.0f;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = batch_input[i] - max_val[0];
        float exp_val = expf(val);
        atomicAdd(sum_exp, exp_val);
        batch_output[i] = val;  // Store (x_i - max_val) for later use
    }
    __syncthreads();
    
    // Step 3: Compute log(exp(x_i - max_val)/sum_exp) = (x_i - max_val) - log(sum_exp)
    float log_sum_exp = logf(sum_exp[0]);
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        batch_output[i] -= log_sum_exp;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    // Input validation
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(dim == 0 || dim == 1, "Only dim 0 or 1 supported");
    
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int dim_size = sizes[1];
    
    if (dim == 0) {
        // Transpose the operation if dim=0
        batch_size = sizes[1];
        dim_size = sizes[0];
    }
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int shared_mem_size = 2 * sizeof(float);  // For max_val and sum_exp
    
    log_softmax_kernel<<<batch_size, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_size,
        batch_size,
        dim
    );
    
    if (dim == 0) {
        output = output.t();
    }
    
    return output;
}
"""

log_softmax_cpp_source = "torch::Tensor log_softmax_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
log_softmax = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using custom CUDA kernel.
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
        return log_softmax.log_softmax_cuda(x, self.dim)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()  # Move to GPU for CUDA kernel
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed