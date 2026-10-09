#pragma once
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mini {

//Used to store float numbers in a multi-dimensional array.
class Tensor {
    public: 
        Tensor() = default; 
        explicit Tensor(std::vector<size_t> shape) 
            : shape_(std::move(shape)), data_(numel_of(shape_), 0.0f) {} 
        Tensor(std::vector<size_t> shape, std::vector<float> data) 
            : shape_(std::move(shape)), data_(std::move(data)) { 
        if (data_.size() != numel_of(shape_)) 
            throw std::invalid_argument("Tensor: data size != shape product"); 
        } 

    const std::vector<size_t>& shape() const { return shape_; } 
    size_t dim(size_t i) const { return shape_.at(i); } 
    size_t ndim() const { return shape_.size(); } 
    size_t numel() const { return data_.size(); } 

    float* data() { return data_.data(); } 
    const float* data() const { return data_.data(); } 

    // 2D access: row-major order 
    float& at(size_t r, size_t c) { return data_[r * shape_[1] + c]; } 
    float at(size_t r, size_t c) const { return data_[r * shape_[1] + c]; }

    private: 
        static size_t numel_of(const std::vector<size_t>& s) { 
            return std::accumulate(s.begin(), s.end(), size_t{1}, 
            [](size_t a, size_t b) { return a * b; }); 
        } 
    std::vector<size_t> shape_; //{2, 3} describes the multidimensional structure.
    std::vector<float> data_;   //{0, 0, 0, 0, 0, 0}; Use one-dimensional contiguous storage.
};

} // namespace mini