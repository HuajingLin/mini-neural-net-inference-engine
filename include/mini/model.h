#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "mini/layers.h"

namespace mini {

// A simple neural network container: Sequential
class Sequential {
public:
    // Construct a layer in place and add it to the end. 
    // such as: model.emplace<Linear>(w, b);
    template <typename L, typename... Args> L& emplace(Args&&... args) {
        //Forward the passed arguments to L's constructor.
        auto p = std::make_unique<L>(std::forward<Args>(args)...);
        L& ref = *p;
        layers_.push_back(std::move(p));
        return ref;
    }

    Tensor forward(Tensor x) const {
        for (const auto& layer : layers_) {
            x = layer->forward(std::move(x));
        }
        return x;
    }

    size_t size() const { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace mini
