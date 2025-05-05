# runtime: 4.22
# basline: 8.71
# speedup: 2.0635628044391856
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the optimized CUDA kernel for Layer Normalization
layer_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 512;

template<typename T>
__device__ __forceinline__ T warp_shfl_down(T val, int offset) {
    return __shfl_down_sync(0xFFFFFFFF, val, offset);
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        val += warp_shfl_down(val, offset);
    return val;
}

template<typename scalar_t, int THREADS_PER_BLOCK>
__global__ void layer_norm_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int num_elements,
    const int num_features,
    const float eps) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const scalar_t* current_input = input + batch_idx * num_elements;
    scalar_t* current_output = output + batch_idx * num_elements;
    
    __shared__ float s_mean;
    __shared__ float s_var;
    __shared__ float s_inv_std;
    
    // Combined mean and variance calculation in one pass
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < num_elements; i += THREADS_PER_BLOCK) {
        float val = static_cast<float>(current_input[i]);
        sum += val;
        sum_sq += val * val;
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);
    
    // Block-level reduction
    if (THREADS_PER_BLOCK > WARP_SIZE) {
        __shared__ float warp_sums[32];
        __shared__ float warp_sums_sq[32];
        
        const int lane = tid % WARP_SIZE;
        const int warp_id = tid / WARP_SIZE;
        
        if (lane == 0) {
            warp_sums[warp_id] = sum;
            warp_sums_sq[warp_id] = sum_sq;
        }
        __syncthreads();
        
        if (warp_id == 0 && lane < (THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE) {
            sum = warp_sums[lane];
            sum_sq = warp_sums_sq[lane];
        } else {
            sum = 0.0f;
            sum_sq = 0.0f;
        }
        
        sum = warp_reduce_sum(sum);
        sum_sq = warp_reduce_sum(sum_sq);
    }
    
    // Final mean and variance calculation
    if (tid == 0) {
        float mean = sum / num_elements;
        float variance = (sum_sq / num_elements) - (mean * mean);
        s_mean = mean;
        s_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    // Normalize and apply affine transformation
    float mean = s_mean;
    float inv_std = s_inv_std;
    
    #pragma unroll 4
    for (int i = tid; i < num_elements; i += THREADS_PER_BLOCK) {
        int feature_idx = i % num_features;
        float val = static_cast<float>(current_input[i]);
        float normalized = (val - mean) * inv_std;
        scalar_t current_weight = weight ? weight[feature_idx] : scalar_t(1.0);
        scalar_t current_bias = bias ? bias[feature_idx] : scalar_t(0.0);
        current_output[i] = static_cast<scalar_t>(normalized) * current_weight + current_bias;
    }
}

torch::Tensor layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_elements,
    int num_features,
    float eps) {
    
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    dim3 blocks(batch_size);
    
    // Optimal thread count based on problem size
    int threads = min(MAX_THREADS, ((num_elements + 31) / 32) * 32);
    threads = max(WARP_SIZE, threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "layer_norm_cuda", [&] {
        const scalar_t* input_data = input.data_ptr<scalar_t>();
        const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
        const scalar_t* bias_data = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
        scalar_t* output_data = output.data_ptr<scalar_t>();
        
        switch(threads) {
            case 32:
                layer_norm_kernel<scalar_t, 32><<<blocks, threads>>>(
                    input_data, weight_data, bias_data, output_data, 
                    num_elements, num_features, eps);
                break;
            case 64:
                layer_norm_kernel<scalar_t, 64><<<blocks, threads>>>(
                    input_data, weight_data, bias_data, output_data, 
                    num_elements, num_features, eps);
                break;
            case 128:
                layer_norm_kernel<scalar_t, 128><<<blocks, threads>>>(
                    input_data, weight_data, bias_data, output_data, 
                    num_elements, num_features, eps);
                break;
            case 256:
                layer_norm_kernel<scalar_t, 256><<<blocks, threads>>>(
                    input_data, weight_data, bias_data, output_data, 
                    num_elements, num_features, eps);
                break;
            case 512:
                layer_norm_kernel<scalar_t, 512><<<blocks, threads>>>(
                    input_data, weight_data, bias_data, output_data, 
                    num_elements, num_features, eps);
                break;
            default:
                layer_norm_kernel<scalar_t, 256><<<blocks, 256>>>(
                    input_data, weight_data, bias_data, output_data, 
                    num_elements, num_features, eps);
                break;
        }
    });
    
    return output;
}
"""

layer_norm_cpp_source = """
torch::Tensor layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_elements,
    int num_features,
    float eps);
"""

# Compile the inline CUDA code
layer_norm_cuda = load_inline(
    name="layer_norm_cuda",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    extra_cuda_cflags=["-DCUDA_HAS_FP16=1", "--fmad=true", "-O3", "--expt-relaxed-constexpr"]
)

class ModelNew(nn.Module):
    """
    Optimized implementation of Layer Normalization with CUDA kernels.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.num_elements = 1
        for s in normalized_shape:
            self.num_elements *= s
            
        self.num_features = normalized_shape[-1] if len(normalized_shape) > 0 else 1
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten all dimensions except batch and last normalized dimension
        original_shape = x.shape
        x = x.contiguous().view(-1, self.num_elements)
        
        output = layer_norm_cuda.layer_norm_cuda(
            x,
            self.weight,
            self.bias,
            self.num_elements,
            self.num_features,
            self.eps
        )
        
        # Restore original shape
        return output.view(original_shape)