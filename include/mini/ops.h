#pragma once
#include <cstddef>
#include "mini/tensor.h"

namespace mini {

// C[M,N] = A[M,K] * B[K,N], row-major order; C will be overwritten
void gemm(const float* A, const float* B, float* C,
            size_t M, size_t K, size_t N);

// Tensor version: a:[M,K]  b:[K,N]  ->  [M,N]; performs shape validation
Tensor matmul(const Tensor& a, const Tensor& b);

//In-place operation: Add bias[N] to each row of X[M, N].
void add_bias(float* X, const float* bias, size_t M, size_t N);

//In-place operation: ReLU(x) = max(0, x)
void relu(float* X, size_t n);

//In-place operation: Apply numerically stable softmax to each row of X[M, N].
void softmax(float* X, size_t M, size_t N);

}  // namespace mini