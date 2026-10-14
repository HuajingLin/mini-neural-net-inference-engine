#pragma once
#include <string>
#include <unordered_map>

#include "mini/model.h"
#include "mini/tensor.h"

namespace mini {

using TensorMap = std::unordered_map<std::string, Tensor>;

// Read MINI format files (assuming the machine is little endian, x86 / ARM are all satisfied)
TensorMap load_tensors(const std::string& path);

// Use the weights of fc1..fc3 to build an MLP of 784-128-64-10, and the output is logits (excluding softmax)
Sequential build_mlp(const TensorMap& weights);

} // namespace mini