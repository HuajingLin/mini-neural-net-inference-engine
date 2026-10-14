#include "mini/loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini {

static_assert(sizeof(float) == 4, "float must be 32-bit");

template <typename T>
static T read_pod(std::ifstream& in) {
    T v;
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!in)
        throw std::runtime_error("load_tensors: unexpected end of file");
    return v;
}

// Load a set of named tensors from a MINI format file
TensorMap load_tensors(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("load_tensors: cannot open " + path);

    char magic[4];
    in.read(magic, 4);
    if (!in || std::memcmp(magic, "MINI", 4) != 0)
        throw std::runtime_error("load_tensors: bad magic in " + path);
    if (read_pod<uint32_t>(in) != 1)
        throw std::runtime_error("load_tensors: unsupported version");

    const uint32_t count = read_pod<uint32_t>(in); 
    TensorMap out; 
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t name_len = read_pod<uint32_t>(in);
        if (name_len == 0 || name_len > 256)
            throw std::runtime_error("load_tensors: bad name length");
        std::string name(name_len, '\0');
        in.read(&name[0], name_len);

        const uint32_t ndim = read_pod<uint32_t>(in);
        if (ndim == 0 || ndim > 8)
            throw std::runtime_error("load_tensors: bad ndim");
        std::vector<size_t> shape(ndim);
        size_t numel = 1;
        for (auto& d : shape) {
            d = read_pod<uint32_t>(in);
            numel *= d;
        }

        std::vector<float> data(numel);
        in.read(reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(numel * sizeof(float)));
        if (!in)
            throw std::runtime_error("load_tensors: truncated data for " + name);

        out.emplace(std::move(name), Tensor(std::move(shape), std::move(data)));
    } 
    return out;
}

// Build a simple MLP with 3 fully connected layers and ReLU activations
Sequential build_mlp(const TensorMap& weights) { 
    Sequential model; 
    for (int i = 1; i <= 3; ++i) { 
        const std::string prefix = "fc" + std::to_string(i); 
        model.emplace<Linear>(weights.at(prefix + ".weight"), 
        weights.at(prefix + ".bias"));
        if (i < 3)
            model.emplace<ReLU>(); 
    } 
    return model;
}

} // namespace mini