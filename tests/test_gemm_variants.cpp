#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "mini/ops.h"

#define CHECK(cond) do { if (!(cond)) { \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); std::exit(1);} } while (0)

struct Variant { const char* name; mini::GemmFn fn; };

int main() {
    const Variant variants[] = {
        {"ikj", mini::gemm_ikj},
        {"blocked", mini::gemm_blocked},
    };
    const size_t shapes[][3] = {
        {1, 1, 1}, {2, 3, 2}, {3, 5, 7}, {17, 100, 33},
        {5, 65, 129}, {64, 784, 128}, {1, 784, 128},
    };

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const auto& s : shapes) {
        const size_t M = s[0], K = s[1], N = s[2];
        std::vector<float> A(M * K), B(K * N), ref(M * N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);
        mini::gemm_naive(A.data(), B.data(), ref.data(), M, K, N);

        for (const auto& v : variants) {
            std::vector<float> C(M * N, 123.0f); //Garbage initial value
            v.fn(A.data(), B.data(), C.data(), M, K, N);

            float max_err = 0.0f;
            for (size_t i = 0; i < C.size(); ++i)
                max_err = std::fmax(max_err, std::fabs(C[i] - ref[i]));
            std::printf("%-8s %4zux%4zux%4zu  max abs err vs naive: %g\n",
                        v.name, M, K, N, max_err);
            CHECK(max_err < 1e-4f);
        }
    }
    std::puts("gemm variants ok");
}
