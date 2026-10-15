#include "mini/mnist.h"

#include <fstream>
#include <stdexcept>

namespace mini {

// Big-Endian to Little-Endian
static uint32_t read_be32(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in)
        throw std::runtime_error("mnist: unexpected end of file");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
            (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

MnistImages load_mnist_images(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("mnist: cannot open " + path);
    if (read_be32(in) != 0x00000803)
        throw std::runtime_error("mnist: bad magic (not an IDX image file): " + path);

    MnistImages im;
    im.count = read_be32(in);
    im.rows = read_be32(in);
    im.cols = read_be32(in);
    if (im.rows != 28 || im.cols != 28 || im.count == 0 || im.count > 100000)
        throw std::runtime_error("mnist: unexpected image dimensions");

    im.pixels.resize(im.count * im.rows * im.cols);
    in.read(reinterpret_cast<char*>(im.pixels.data()),
    static_cast<std::streamsize>(im.pixels.size()));
    if (!in)
        throw std::runtime_error("mnist: truncated image data");
    return im;
}

std::vector<uint8_t> load_mnist_labels(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("mnist: cannot open " + path);
    if (read_be32(in) != 0x00000801)
        throw std::runtime_error("mnist: bad magic (not an IDX label file): " + path);

    const size_t count = read_be32(in);
    if (count == 0 || count > 100000)
        throw std::runtime_error("mnist: unexpected label count");

    std::vector<uint8_t> labels(count);
    in.read(reinterpret_cast<char*>(labels.data()),
            static_cast<std::streamsize>(count));
    if (!in)
        throw std::runtime_error("mnist: truncated label data");
    return labels;
}

Tensor mnist_batch(const MnistImages& im, size_t begin, size_t n) {
    if (begin + n > im.count)
        throw std::out_of_range("mnist_batch: range out of bounds");

    const size_t dim = im.rows * im.cols;
    Tensor x({n, dim});
    const uint8_t* src = im.pixels.data() + begin * dim;
    float* dst = x.data();
    for (size_t i = 0; i < n * dim; ++i) {
        // On the Python side, `ToTensor` and `Normalize((0.1307,), (0.3081,))` are used.
        // x = (pixel / 255 - 0.1307) / 0.3081
        dst[i] = (src[i] / 255.0f - 0.1307f) / 0.3081f;
    }
    return x;
}

} // namespace mini