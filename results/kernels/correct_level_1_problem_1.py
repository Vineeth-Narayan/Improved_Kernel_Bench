# runtime: 24.4
# basline: 4.01
# speedup: 0.16434426229508198
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled matrix multiplication kernel with shared memory
template <int TILE_SIZE>
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < N; t += TILE_SIZE) {
        // Load tile from A into shared memory
        if (row < N && (t + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        if ((t + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result to output
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    // Using 16x16 tiles (can be tuned for specific hardware)
    const int TILE_SIZE = 16;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<TILE_SIZE><<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
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
        """
        Performs the matrix multiplication using custom CUDA kernel.
        
        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).
            
        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N).cuda()
    B = torch.randn(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed