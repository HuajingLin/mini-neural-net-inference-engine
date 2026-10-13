#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "mini/model.h"

#define CHECK(cond) do { if (!(cond)) { \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); std::exit(1);} } while (0)

using namespace mini;

static void test_hand_computed() {
    Sequential model;
    model.emplace<Linear>(Tensor({2, 2}, {1, -1, 2, 0}), Tensor({2}, {0, -1}));
    model.emplace<ReLU>();
    model.emplace<Linear>(Tensor({2, 1}, {1, 3}), Tensor({1}, {-1}));

    Tensor y = model.forward(Tensor({2, 2}, {1, 2, -1, 0}));    //input: [[1,2],[-1,0]]
    CHECK(y.dim(0) == 2 && y.dim(1) == 1);
    CHECK(y.at(0, 0) == 4.0f);
    CHECK(y.at(1, 0) == -1.0f);
    std::puts("hand-computed ok");
}

static void test_mnist_shape() {
    //All-zero weights and biases: logits all 0, 
    //After the softmax operation, the probability for each class should be 0.1.
    Sequential model;
    model.emplace<Linear>(Tensor({784, 128}), Tensor({128}));
    model.emplace<ReLU>();
    model.emplace<Linear>(Tensor({128, 64}), Tensor({64}));
    model.emplace<ReLU>();
    model.emplace<Linear>(Tensor({64, 10}), Tensor({10}));
    model.emplace<Softmax>();
    CHECK(model.size() == 6);

    Tensor y = model.forward(Tensor({3, 784}));   // batch = 3
    CHECK(y.dim(0) == 3 && y.dim(1) == 10);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 10; ++j)
            CHECK(std::fabs(y.at(i, j) - 0.1f) < 1e-6f);
    std::puts("mnist shape ok");
}

static void test_shape_errors() {
    Sequential model;
    model.emplace<Linear>(Tensor({4, 2}), Tensor({2}));

    //Test: Incorrect input shape
    bool threw = false;
    try { 
        model.forward(Tensor({1, 5})); // Expected 4 columns, but 5 were provided.
    }     
    catch (const std::invalid_argument&) { 
        threw = true; 
    }
    CHECK(threw);

    //Test: Bias Shape Error
    threw = false;
    try { 
        Linear bad(Tensor({4, 2}), Tensor({3})); // The bias length should be 2.
    }   
    catch (const std::invalid_argument&) { 
        threw = true; 
    }
    CHECK(threw);
    std::puts("shape errors ok");
}

int main() {
    test_hand_computed();
    test_mnist_shape();
    test_shape_errors();
}
