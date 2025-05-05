# runtime: 18.5
# baseline: 4.01
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the optimized CUDA kernel for matrix multiplication
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_K = 8;

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < N; m += BLOCK_SIZE) {
        // Load tiles into shared memory with bounds checking
        if (row < N && (m + tx) < N) {
            As[ty][tx] = A[row * N + (m + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((m + ty) < N && col < N) {
            Bs[ty][tx] = B[(m + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory with bounds checking
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-arch=sm_75", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N, dtype=torch.float32).cuda()
    B = torch.randn(N, N, dtype=torch.float32).cuda()
    return [A, B]

def get_init_inputs():
    return []