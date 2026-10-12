#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "mini/ops.h"

#define CHECK(cond) do { if (!(cond)) { \
std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); std::exit(1);} } while (0)

static void test_hand_computed() { 
    mini::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6}); 
    mini::Tensor b({3, 2}, {7, 8, 9, 10, 11, 12}); 
    mini::Tensor c = mini::matmul(a, b); 
    CHECK(c.dim(0) == 2 && c.dim(1) == 2); //row and column dimensions
    CHECK(c.at(0, 0) == 58 && c.at(0, 1) == 64); 
    CHECK(c.at(1, 0) == 139 && c.at(1, 1) == 154); 
    std::puts("hand-computed ok");
}

static std::vector<float> read_vec(std::ifstream& in, size_t n) { 
    std::vector<float> v(n); 
    for (auto& x : v) in >> x; 
    return v;
}

static void test_against_numpy(const char* path) { 
    std::ifstream in(path); 
    if (!in) { 
        std::printf("cannot open %s\n", path); std::exit(1); 
    } 

    int cases = 0; 
    in >> cases; //first line is the number of test cases
    for (int t = 0; t < cases; ++t) { 
        size_t M, K, N; 
        in >> M >> K >> N; 
        auto A = read_vec(in, M * K); 
        auto B = read_vec(in, K * N); 
        auto ref = read_vec(in, M * N); 

        std::vector<float> C(M * N); 
        mini::gemm(A.data(), B.data(), C.data(), M, K, N); 

        float max_err = 0.0f; 
        for (size_t i = 0; i < C.size(); ++i) 
            max_err = std::fmax(max_err, std::fabs(C[i] - ref[i])); 
        std::printf("case %d (%zux%zux%zu): max abs err = %g\n", t, M, K, N, max_err); 
        CHECK(max_err < 1e-3f); 
    } 
    std::puts("numpy comparison ok");
}

int main(int argc, char** argv) { 
    test_hand_computed(); 
    if (argc > 1) test_against_numpy(argv[1]);
}