#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <numeric>

#include "logger.h"

template <typename DataType>
    requires std::integral<DataType> || std::floating_point<DataType>
class Tensor {
public:
    Tensor()
        : data_(nullptr)
        , n_dims(1)
    {
        std::fill(shape_, shape_ + max_dim, 0);
    }

    Tensor(const DataType* data, const std::initializer_list<size_t>& shape)
    {
        if (shape.size() > max_dim)
            throw std::invalid_argument("Tensor shape cannot have more than 4 dimensions");

        n_dims = shape.size();
        std::copy(shape.begin(), shape.end(), shape_);
        for (size_t i = shape.size(); i < max_dim; i++)
            shape_[i] = 0;

        size_t size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());
        data_ = new DataType[size];

        if (data != nullptr)
            std::copy(data, data + size, data_);
        else
            std::fill(data_, data_ + size, 0);
    }

    const DataType* flat() const { return data_; }

    size_t size() const
    {
        size_t size = 1;
        for (size_t i = 0; i < n_dims; i++)
            size *= shape_[i];
        return size;
    }

    DataType operator[](std::initializer_list<size_t> indices) const
    {
        if (indices.size() != n_dims)
            throw std::invalid_argument("Number of indices does not match the number of dimensions");
        return data_[multi_indices_to_flat(indices)];
    }

    const size_t* shape() const { return shape_; }

    ~Tensor() { delete data_; }

private:
    size_t multi_indices_to_flat(std::initializer_list<size_t> indices) const
    {
        size_t flat_idx = 0;
        size_t stride = 1;
        for (int i = n_dims - 1; i >= 0; i--) {
            if (indices.begin()[i] >= shape_[i])
                throw std::out_of_range("Index out of bounds");
            flat_idx += indices.begin()[i] * stride;
            stride *= shape_[i];
        }
        return flat_idx;
    }

private:
    static constexpr size_t max_dim = 4;
    size_t shape_[max_dim];
    size_t n_dims;
    DataType* data_;
};
