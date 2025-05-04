import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA + C++ WMMA kernel for 16x16x16 tile matmul C = A * B
source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define M 16
#define N 16
#define K 16

__global__ void wmma_matmul_kernel(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // Initialize output to 0
    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrices into fragments
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

torch::Tensor wmma_matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kHalf, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kHalf, "B must be float16");

    auto C = torch::zeros({16, 16}, torch::dtype(torch::kFloat).device(A.device()));

    // Launch one warp (since this is just one tile)
    dim3 blockDim(32);
    dim3 gridDim(1);
    wmma_matmul_kernel<<<gridDim, blockDim>>>(
        reinterpret_cast<half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>());

    return C;
}
"""

cpp_src = ("torch::Tensor wmma_matmul(torch::Tensor A, torch::Tensor B);")

wmma_module = load_inline(
    name="wmma_matmul_module",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["wmma_matmul"],
    verbose=True
)

# PyTorch wrapper
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = wmma_module

    def forward(self, A, B):
        return self.matmul.wmma_matmul(A, B)
