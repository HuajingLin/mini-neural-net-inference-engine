#include "mini/ops.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>

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

void add_bias(float* X, const float* bias, size_t M, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        float* row = X + i * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] += bias[j];
        }
    }
}

void relu(float* X, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (X[i] < 0.0f) X[i] = 0.0f;
    }
}

void softmax(float* X, size_t M, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        float* row = X + i * N;

        float max_val = row[0];
        for (size_t j = 1; j < N; ++j) max_val = std::max(max_val, row[j]);

        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }

        const float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < N; ++j) row[j] *= inv_sum;
    }
}

} // namespace mini