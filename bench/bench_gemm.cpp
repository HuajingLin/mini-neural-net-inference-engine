#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mini/ops.h"

struct Variant { const char* name; mini::GemmFn fn; };

static double best_seconds(mini::GemmFn fn, const std::vector<float>& A,
                           const std::vector<float>& B, std::vector<float>& C,
                           size_t M, size_t K, size_t N) {
    const double flops = 2.0 * M * K * N;
    const int reps = std::max(1, static_cast<int>(3e8 / flops));
    fn(A.data(), B.data(), C.data(), M, K, N);                 // Warm-up
    double best = 1e30;
    for (int trial = 0; trial < 5; ++trial) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) fn(A.data(), B.data(), C.data(), M, K, N);
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count() / reps);
    }
    return best;
}

int main() {
    const Variant variants[] = {
        {"naive(ijk)", mini::gemm_naive},
        {"ikj", mini::gemm_ikj},
        {"blocked", mini::gemm_blocked},
    };
    const size_t shapes[][3] = {
        {256, 784, 128}, {256, 128, 64}, {256, 64, 10}, {1, 784, 128},
    };

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::printf("%-14s", "MxKxN");
    for (const auto& v : variants) std::printf("%16s", v.name);
    std::printf("   (GFLOPS, higher is better)\n");

    for (const auto& s : shapes) {
        const size_t M = s[0], K = s[1], N = s[2];
        std::vector<float> A(M * K), B(K * N), C(M * N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);

        char label[32];
        std::snprintf(label, sizeof label, "%zux%zux%zu", M, K, N);
        std::printf("%-14s", label);

        double sums[3] = {0, 0, 0};
        size_t vi = 0;
        for (const auto& v : variants) {
            const double sec = best_seconds(v.fn, A, B, C, M, K, N);
            for (float x : C)
                sums[vi] += x;     // comparisons could be made across different versions
            std::printf("%16.2f", 2.0 * M * K * N / sec / 1e9);
            ++vi;
        }
        std::printf("   checksum diff: %.1e\n",
                    std::max(std::fabs(sums[1] - sums[0]), std::fabs(sums[2] - sums[0])));
    }
}
