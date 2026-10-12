#include "mini/ops.h"
#include <stdexcept>

namespace mini {

void gemm(const float* A, const float* B, float* C, 
            size_t M, size_t K, size_t N) { 
    for (size_t i = 0; i < M; ++i) { 
        for (size_t j = 0; j < N; ++j) { 
            float sum = 0.0f; 
            for (size_t k = 0; k < K; ++k) { 
                sum += A[i * K + k] * B[k * N + j]; 
            } 
            C[i * N + j] = sum; 
        } 
    }
}

Tensor matmul(const Tensor& a, const Tensor& b) { 
    if (a.ndim() != 2 || b.ndim() != 2) 
        throw std::invalid_argument("matmul: expects 2D tensors"); 
    if (a.dim(1) != b.dim(0)) 
        throw std::invalid_argument("matmul: inner dimensions mismatch"); 

    const size_t M = a.dim(0), K = a.dim(1), N = b.dim(1); 
    Tensor c({M, N}); 
    gemm(a.data(), b.data(), c.data(), M, K, N); 
    return c;
}

} // namespace mini