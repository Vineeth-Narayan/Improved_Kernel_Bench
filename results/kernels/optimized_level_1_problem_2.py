# runtime: 18.5
# basline: 3.84
# speedup: 0.20756756756756756
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile dimensions
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                              int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of C to compute
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        if (col < N && (t * TILE_SIZE + ty) < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get matrix dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Grid and block dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), 
                                      B.data_ptr<float>(), 
                                      C.data_ptr<float>(), 
                                      M, N, K);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)

M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(M, K).cuda()
    B = torch.randn(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed