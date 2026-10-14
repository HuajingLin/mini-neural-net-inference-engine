#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "mini/loader.h"

#define CHECK(cond) do { if (!(cond)) { \
std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); std::exit(1);} } while (0)

static size_t argmax(const float* p, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i)
    if (p[i] > p[best])
        best = i;
    return best;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::puts("usage: test_weights <weights.bin> <ref.bin>");
        return 1;
    }
    //Load two binary files.
    //mnist_mlp.bin: Model weights
    //mnist_ref.bin: Inputs, labels, and reference logits
    auto weights = mini::load_tensors(argv[1]);
    auto ref = mini::load_tensors(argv[2]);
    auto model = mini::build_mlp(weights);  // Build a neural network

    const mini::Tensor& input = ref.at("input");
    const mini::Tensor& expected = ref.at("logits");
    const mini::Tensor& labels = ref.at("labels");
    const size_t N = input.dim(0);

    //Perform forward propagation
    mini::Tensor logits = model.forward(input);
    CHECK(logits.dim(0) == N && logits.dim(1) == 10);

    // Comparison of Outputs
    // Maximum absolute error, predicted class differences, classification accuracy
    float max_err = 0.0f;
    size_t mismatches = 0, correct = 0;
    for (size_t i = 0; i < N; ++i) {
        const float* mine = logits.data() + i * 10;
        const float* theirs = expected.data() + i * 10;
        for (size_t j = 0; j < 10;++j)
            max_err = std::fmax(max_err, std::fabs(mine[j] - theirs[j]));

        const size_t pred = argmax(mine, 10);
        if (pred != argmax(theirs, 10))
            ++mismatches;
        if (pred == static_cast<size_t>(labels.data()[i]))
            ++correct;
    }

    std::printf("samples: %zu\n", N);
    std::printf("max abs logits error vs PyTorch: %g\n", max_err);
    std::printf("prediction mismatches vs PyTorch: %zu\n", mismatches);
    std::printf("accuracy on these samples: %zu/%zu\n", correct, N);

    CHECK(max_err < 1e-3f);
    CHECK(mismatches == 0);
    std::puts("weights alignment ok");
}