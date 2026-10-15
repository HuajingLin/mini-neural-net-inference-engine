#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <vector>

#include "mini/loader.h"
#include "mini/mnist.h"
#include "mini/ops.h"

using namespace mini;

static void print_digit(const MnistImages& im, size_t idx) {
    const char* shades = " .:-=+*#%@";          // 10 levels of brightness
    const uint8_t* p = im.pixels.data() + idx * im.rows * im.cols;
    for (size_t r = 0; r < im.rows; ++r) {
        for (size_t c = 0; c < im.cols; ++c) {
            const char ch = shades[p[r * im.cols + c] * 9 / 255];
            std::putchar(ch);
            std::putchar(ch);                    // Print two characters per pixel to make the image closer to square
        }
        std::putchar('\n');
    }
}

static void run_single(const Sequential& model, const MnistImages& im,
                       const std::vector<uint8_t>& labels, size_t idx) {
    print_digit(im, idx);

    Tensor logits = model.forward(mnist_batch(im, idx, 1)); // Single-image inference
    const size_t pred = argmax(logits.data(), 10);   // Softmax not needed for predicting the class
    softmax(logits.data(), 1, 10);                    // Calculate only when viewing probabilities

    std::printf("\nimage #%zu   label: %d   prediction: %zu   %s\n", idx,
                labels[idx], pred, pred == labels[idx] ? "(correct)" : "(WRONG)");
    for (size_t j = 0; j < 10; ++j)
        std::printf(" %zu: %.4f\n", j, logits.at(0, j));
}

static void run_eval(const Sequential& model, const MnistImages& im,
                     const std::vector<uint8_t>& labels) {
    const size_t batch = 256;
    size_t correct = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t b = 0; b < im.count; b += batch) {
        const size_t n = std::min(batch, im.count - b);
        Tensor logits = model.forward(mnist_batch(im, b, n));
        for (size_t i = 0; i < n; ++i) {
            if (argmax(logits.data() + i * 10, 10) == labels[b + i])
                ++correct;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::printf("accuracy: %zu/%zu = %.2f%%\n", correct, im.count,
                100.0 * correct / im.count);
    std::printf("time: %.3f s (%.0f images/s, includes preprocessing)\n", sec,
                im.count/sec);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
        "usage: %s <weights.bin> <images-idx3> <labels-idx1> [index]\n",
        argv[0]);
        return 1;
    }
    try {
        Sequential model = build_mlp(load_tensors(argv[1]));    // models/mnist_mlp.bin 
        MnistImages images = load_mnist_images(argv[2]);        // data/t10k-images-idx3-ubyte
        std::vector<uint8_t> labels = load_mnist_labels(argv[3]);// data/t10k-labels-idx1-ubyte
        if (labels.size() != images.count)
            throw std::runtime_error("images/labels count mismatch");

        if (argc >= 5) {
            const size_t idx = std::strtoul(argv[4], nullptr, 10);
            if (idx >= images.count)
                throw std::out_of_range("index out of range");
            run_single(model, images, labels, idx);
        } else {
            run_eval(model, images, labels);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}