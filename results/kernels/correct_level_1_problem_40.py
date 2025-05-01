import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Layer Normalization
layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

template<typename T>
__device__ __inline__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<typename T>
__device__ __inline__ T block_reduce_sum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

template<typename T>
__global__ void layer_norm_forward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    T* __restrict__ mean,
    T* __restrict__ rstd,
    const int num_instances,
    const int num_features,
    const int feature_stride,
    const float eps) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= num_instances) return;
    
    // Compute mean
    T sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        sum += input[bid * feature_stride + i];
    }
    
    sum = block_reduce_sum(sum);
    if (tid == 0) {
        mean[bid] = sum / num_features;
    }
    __syncthreads();
    
    // Compute variance (using the mean)
    T mean_val = mean[bid];
    T var_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        T diff = input[bid * feature_stride + i] - mean_val;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum(var_sum);
    if (tid == 0) {
        rstd[bid] = rsqrtf(var_sum / num_features + eps);
    }
    __syncthreads();
    
    // Normalize and apply weight/bias
    T scale = rstd[bid];
    for (int i = tid; i < num_features; i += blockDim.x) {
        output[bid * feature_stride + i] = 
            (input[bid * feature_stride + i] - mean_val) * scale * weight[i] + bias[i];
    }
}

torch::Tensor layer_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    auto input_shape = input.sizes();
    int num_instances = input.numel() / weight.numel();
    int num_features = weight.numel();
    int feature_stride = weight.numel();
    
    auto output = torch::empty_like(input);
    auto mean = torch::empty({num_instances}, input.options());
    auto rstd = torch::empty({num_instances}, input.options());
    
    const int block_size = 256;
    dim3 grid(num_instances);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "layer_norm_forward_cuda", ([&] {
        layer_norm_forward_kernel<scalar_t><<<grid, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            rstd.data_ptr<scalar_t>(),
            num_instances,
            num_features,
            feature_stride,
            eps);
    }));
    
    return output;
}
"""

layer_norm_cpp_source = """
torch::Tensor layer_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);
"""

# Compile the inline CUDA code
layer_norm_cuda = load_inline(
    name="layer_norm_cuda",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized LayerNorm model with custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        self.layer_norm_cuda = layer_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch::Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        input_shape = x.shape
        normalized_dims = len(self.normalized_shape)
        num_features = torch.prod(torch.tensor(self.normalized_shape)).item()
        
        # Flatten all dimensions except the normalized ones
        x = x.contiguous().view(-1, num_features)
        
        # Apply custom layer norm
        x = self.layer_norm_cuda.layer_norm_forward_cuda(x, 
            self.weight.view(-1), 
            self.bias.view(-1), 
            self.eps)
        
        # Restore original shape
        return x.view(input_shape)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]