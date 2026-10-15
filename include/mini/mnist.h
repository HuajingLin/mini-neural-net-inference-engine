// mnist.h:
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "mini/tensor.h"

namespace mini {

struct MnistImages {
    size_t count = 0, rows = 0, cols = 0;
    std::vector<uint8_t> pixels;   // count * rows * cols, values ​​in range [0, 255]
};

MnistImages load_mnist_images(const std::string& path);
std::vector<uint8_t> load_mnist_labels(const std::string& path);

// Selects images in the range [begin, begin+n), applies the same normalization
// used during training, and returns a tensor of shape [n, rows*cols]
Tensor mnist_batch(const MnistImages& images, size_t begin, size_t n);

}  // namespace mini