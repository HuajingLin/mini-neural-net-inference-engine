#include "mini/ops.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace mini {

void gemm_naive(const float* A, const float* B, float* C, 
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

void gemm_ikj(const float* A, const float* B, float* C,
              size_t M, size_t K, size_t N) {
    std::fill(C, C + M * N, 0.0f);// C is the accumulation target and must be cleared to zero first
    for (size_t i = 0; i < M; ++i) {
        float* c_row = C + i * N;
        for (size_t k = 0; k < K; ++k) {
            const float a = A[i * K + k];
            const float* b_row = B + k * N;
            for (size_t j = 0; j < N; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

// Blocking: Keep a small KC * NC block of B in the cache so it can be reused by all rows of A.
void gemm_blocked(const float* A, const float* B, float* C,
                  size_t M, size_t K, size_t N) {
    constexpr size_t KC = 64;
    constexpr size_t NC = 128;               // 64*128*4 bytes = 32 KB, one L1 data cache

    std::fill(C, C + M * N, 0.0f);
    for (size_t jj = 0; jj < N; jj += NC) {
        const size_t j_end = std::min(jj + NC, N);
        for (size_t kk = 0; kk < K; kk += KC) {
            const size_t k_end = std::min(kk + KC, K);
            for (size_t i = 0; i < M; ++i) {
                float* c_row = C + i * N;
                for (size_t k = kk; k < k_end; ++k) {
                    const float a = A[i * K + k];
                    const float* b_row = B + k * N;
                    for (size_t j = jj; j < j_end; ++j) {
                        c_row[j] += a * b_row[j];
                    }
                }
            }
        }
    }
}

void gemm(const float* A, const float* B, float* C,
          size_t M, size_t K, size_t N) {
    // After reviewing the benchmarks, switch to the faster one.
    gemm_ikj(A, B, C, M, K, N);
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
        for (size_t j = 1; j < N; ++j)
            max_val = std::max(max_val, row[j]);

        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }

        const float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < N; ++j)
            row[j] *= inv_sum;
    }
}

size_t argmax(const float* x, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > x[best])
            best = i;
    }
    return best;
}

} // namespace mini