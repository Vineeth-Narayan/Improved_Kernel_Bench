# runtime: 1050.0
# basline: 8.71
# speedup: 0.008295238095238097
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Layer Normalization
layer_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void layer_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    const int num_elements,
    const int norm_size,
    const float eps) {
    
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_var = &s_data[blockDim.x];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx >= num_elements / norm_size) return;
    
    const scalar_t* slice_input = input + idx * norm_size;
    scalar_t* slice_output = output + idx * norm_size;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < norm_size; ++i) {
        sum += static_cast<float>(slice_input[i]);
    }
    float mean = sum / norm_size;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < norm_size; ++i) {
        float diff = static_cast<float>(slice_input[i]) - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrtf(var_sum / norm_size + eps);
    
    // Normalize and apply affine transformation
    for (int i = 0; i < norm_size; ++i) {
        float normalized = (static_cast<float>(slice_input[i]) - mean) * inv_std;
        slice_output[i] = static_cast<scalar_t>(normalized * static_cast<float>(gamma[i]) + static_cast<float>(beta[i]));
    }
}

torch::Tensor layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps) {
    
    auto output = torch::empty_like(input);
    const auto norm_size = gamma.numel();
    const auto num_slices = input.numel() / norm_size;
    
    const int threads = 256;
    const int blocks = (num_slices + threads - 1) / threads;
    const int shared_mem = 2 * threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "layer_norm_cuda", ([&] {
        layer_norm_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            input.numel(),
            norm_size,
            eps);
    }));
    
    return output;
}
"""

layer_norm_cpp_source = """
torch::Tensor layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps);
"""

# Compile the inline CUDA code
layer_norm_ext = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernels.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with custom CUDA implementation.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        
        # Initialize learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
        self.layer_norm_cuda = layer_norm_ext.layer_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch::Tensor: Output tensor with Layer Normalization applied.
        """
        # Reshape gamma and beta to match the normalized dimensions
        gamma = self.gamma
        beta = self.beta
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(0)
            beta = beta.unsqueeze(0)
            
        return self.layer_norm_cuda(x, gamma, beta, self.eps)