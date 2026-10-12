#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "mini/ops.h"

#define CHECK(cond) do { if (!(cond)) { \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); std::exit(1);} } while (0)

static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

static void test_hand_computed() {
    // bias
    float x[] = {1, 2, 3, 4, 5, 6};          // [2,3]
    float b[] = {10, 20, 30};
    mini::add_bias(x, b, 2, 3);
    CHECK(x[0] == 11 && x[2] == 33 && x[3] == 14 && x[5] == 36);

    // relu
    float r[] = {-1.5f, 0.0f, 2.0f, -0.0f, 7.0f};
    mini::relu(r, 5);
    CHECK(r[0] == 0.0f && r[2] == 2.0f && r[4] == 7.0f);

    // softmax: 均匀输入 -> 均匀输出
    float s[] = {1, 1, 1, 1};
    mini::softmax(s, 1, 4);
    for (int i = 0; i < 4; ++i) CHECK(near(s[i], 0.25f, 1e-6f));

    // softmax: 大数值不应溢出（朴素实现会得到 NaN）
    float big[] = {1000.0f, 1001.0f, 1002.0f};
    mini::softmax(big, 1, 3);
    float sum = big[0] + big[1] + big[2];
    CHECK(!std::isnan(sum) && near(sum, 1.0f, 1e-6f));
    CHECK(big[2] > big[1] && big[1] > big[0]);

    // softmax: 每行和为 1，且各行独立
    float two[] = {0, 0, 100, 0};            // [2,2]
    mini::softmax(two, 2, 2);
    CHECK(near(two[0], 0.5f, 1e-6f) && near(two[1], 0.5f, 1e-6f));
    CHECK(near(two[2] + two[3], 1.0f, 1e-6f) && two[2] > 0.999f);

    std::puts("hand-computed ok");
}

static std::vector<float> read_vec(std::ifstream& in, size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) in >> x;
    return v;
}

static float max_err(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) e = std::fmax(e, std::fabs(a[i] - b[i]));
    return e;
}

static void test_against_numpy(const char* path) {
    std::ifstream in(path);
    if (!in) { std::printf("cannot open %s\n", path); std::exit(1); }

    int cases = 0;
    in >> cases;
    for (int t = 0; t < cases; ++t) {
        size_t M, N;
        in >> M >> N;
        auto X = read_vec(in, M * N);
        auto b = read_vec(in, N);
        auto ref_bias = read_vec(in, M * N);
        auto ref_relu = read_vec(in, M * N);
        auto ref_soft = read_vec(in, M * N);

        auto y = X;
        mini::add_bias(y.data(), b.data(), M, N);
        float e1 = max_err(y, ref_bias);

        y = X;
        mini::relu(y.data(), y.size());
        float e2 = max_err(y, ref_relu);

        y = X;
        mini::softmax(y.data(), M, N);
        float e3 = max_err(y, ref_soft);

        std::printf("case %d (%zux%zu): bias %g  relu %g  softmax %g\n", t, M, N, e1, e2, e3);
        CHECK(e1 < 1e-5f && e2 == 0.0f && e3 < 1e-6f);
    }
    std::puts("numpy comparison ok");
}

int main(int argc, char** argv) {
    test_hand_computed();
    if (argc > 1) test_against_numpy(argv[1]);
}
