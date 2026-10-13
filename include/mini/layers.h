#pragma once
#include "mini/tensor.h"

namespace mini {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(Tensor x) const = 0;
    virtual const char* name() const = 0;
};

// y = x @ W + b      x:[M,in]  W:[in,out]  b:[out]
class Linear : public Layer {
public:
    Linear(Tensor weight, Tensor bias);
    Tensor forward(Tensor x) const override;
    const char* name() const override { return "Linear"; }
    size_t in_features() const { return weight_.dim(0); }
    size_t out_features() const { return weight_.dim(1); }

private:
    Tensor weight_;
    Tensor bias_;
};

class ReLU : public Layer {
public:
    Tensor forward(Tensor x) const override;
    const char* name() const override { return "ReLU"; }
};

// each row of [M,N] do softmax
class Softmax : public Layer {
public:
    Tensor forward(Tensor x) const override;
    const char* name() const override { return "Softmax"; }
};

}  // namespace mini
