#pragma once
#include <cstddef>
#include "mini/tensor.h"

namespace mini {

// C[M,N] = A[M,K] * B[K,N], row-major order; C will be overwritten
void gemm(const float* A, const float* B, float* C,
            size_t M, size_t K, size_t N);

// Tensor version: a:[M,K]  b:[K,N]  ->  [M,N]; performs shape validation
Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace mini