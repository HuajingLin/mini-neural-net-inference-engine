#include "mini/layers.h"

#include <stdexcept>
#include <utility>

#include "mini/ops.h"

namespace mini {

Linear::Linear(Tensor weight, Tensor bias)
    : weight_(std::move(weight)), bias_(std::move(bias)) {
    if (weight_.ndim() != 2)
        throw std::invalid_argument("Linear: weight must be 2D [in,out]");
        
    if (bias_.ndim() != 1 || bias_.dim(0) != weight_.dim(1))
        throw std::invalid_argument("Linear: bias must be 1D [out]");
}

Tensor Linear::forward(Tensor x) const {
    if (x.ndim() != 2 || x.dim(1) != in_features())
        throw std::invalid_argument("Linear: input must be [M,in_features]");

    Tensor y = matmul(x, weight_);
    add_bias(y.data(), bias_.data(), y.dim(0), y.dim(1));
    return y;
}

Tensor ReLU::forward(Tensor x) const {
    relu(x.data(), x.numel());
    return x;
}

Tensor Softmax::forward(Tensor x) const {
    if (x.ndim() != 2)
        throw std::invalid_argument("Softmax: input must be 2D [M,N]");
    softmax(x.data(), x.dim(0), x.dim(1));
    return x;
}

}  // namespace mini
