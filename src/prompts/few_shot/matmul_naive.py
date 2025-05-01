import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# performs a naive matmul C = A * B
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 blockDim(N, M);
    matmul_naive_kernel<<<1, blockDim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

cpp_src = (
    "torch::Tensor matmul(torch::Tensor A, torch::Tensor B);"
)


matmul_module = load_inline(
    name="naive_matmul_module",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_module

    def forward(self, A, B):
        return self.matmul.matmul(A, B)
