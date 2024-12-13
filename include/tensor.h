#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <utility>

#include "logger.h"

template <typename DataType>
    requires std::integral<DataType> || std::floating_point<DataType>
class Tensor {
public:
    Tensor()
        : data_(nullptr)
        , parent(nullptr)
        , n_dims_(1)
        , offset_(0)
    {
        std::fill(shape_, shape_ + MAX_DIM, 0);
        std::fill(strides_, strides_ + MAX_DIM, 0);
    }

    Tensor(DataType* data, const std::vector<size_t>& shape)
        : parent(nullptr)
        , n_dims_(0)
        , offset_(0)
        , data_(data)
        , shape_ { 0 }
    {
        if (shape.size() > MAX_DIM)
            throw std::invalid_argument("Tensor shape cannot have more than 4 dimensions");

        n_dims_ = shape.size();
        for (int i = shape.size() - 1; i >= 0; i--)
            if (shape[i] == 0)
                n_dims_--;

        std::copy(shape.begin(), shape.end(), shape_);
        for (size_t i = shape.size(); i < MAX_DIM; i++)
            shape_[i] = 0;

        init_strides();
    }

    Tensor(DataType* data, size_t shape[Tensor::MAX_DIM])
        : Tensor(data, std::vector<size_t>(shape, shape + MAX_DIM))
    {
    }

    const DataType* flat() const { return data_; }

    size_t size() const
    {
        size_t size = 1;
        for (size_t i = 0; i < n_dims_; i++)
            size *= shape_[i];
        return size;
    }

    DataType operator[](const std::vector<size_t>& indices) const
    {
        if (indices.size() != n_dims_)
            throw std::invalid_argument("Number of indices does not match the number of dimensions");
        return data_[multi_indices_to_flat(indices)];
    }

    Tensor<DataType> operator[](const std::vector<std::pair<size_t, size_t>>& indices)
    {
        if (indices.size() != n_dims_)
            throw std::invalid_argument("Number of indices does not match the number of dimensions");

        std::vector<size_t> begins, ends;
        for (auto& [begin, end] : indices) {
            begins.push_back(begin);
            ends.push_back(end);
        }

        size_t new_shape[MAX_DIM];
        for (size_t i = 0; i < MAX_DIM; i++)
            new_shape[i] = (i < n_dims_) ? (ends[i] - begins[i]) : 0;

        Tensor<DataType> sub_tensor(data_, new_shape);
        size_t new_offset = multi_indices_to_flat(begins) + offset_;
        sub_tensor.offset_ = new_offset;

        if (parent == nullptr)
            sub_tensor.parent = this;
        else
            sub_tensor.parent = parent;

        for (size_t i = 0; i < MAX_DIM; i++)
            sub_tensor.strides_[i] = (i < n_dims_) ? strides_[i] : 0;

        return sub_tensor;
    }

    void init_strides()
    {
        strides_[n_dims_ - 1] = 1;
        for (int i = n_dims_ - 2; i >= 0; i--)
            strides_[i] = strides_[i + 1] * shape_[i + 1];

        for (int i = MAX_DIM - 1; i >= (int)n_dims_; i--)
            strides_[i] = 0;
    }

    std::string flat_string() const
    {
        std::string str = "[";
        for (size_t i = 0; i < ((shape_[0] == 0) ? 1 : shape_[0]) && !is_empty(); i++) {
            for (size_t j = 0; j < ((shape_[1] == 0) ? 1 : shape_[1]); j++) {
                for (size_t k = 0; k < ((shape_[2] == 0) ? 1 : shape_[2]); k++) {
                    for (size_t l = 0; l < ((shape_[3] == 0) ? 1 : shape_[3]); l++) {
                        size_t flat_idx = multi_indices_to_flat({ i, j, k, l });
                        str += std::to_string(data_[flat_idx]) + ", ";
                    }
                }
            }
        }
        str += "]";
        return str;
    }

    bool is_empty() const { return n_dims_ == 0; }

    const size_t* shape() const { return shape_; }

    ~Tensor()
    {
        if (parent == nullptr) {
            delete[] data_;
        }
    }

private:
    size_t multi_indices_to_flat(const std::vector<size_t>& indices) const
    {
        size_t flat_idx = 0;
        for (int i = n_dims_ - 1; i >= 0; i--) {
            if (indices[i] >= shape_[i] && (shape_[i] != 0 || indices[i] != 0)) {
                throw std::out_of_range("Index out of bounds");
            }
            flat_idx += indices[i] * strides_[i];
        }
        return flat_idx + offset_;
    }

    std::string shape_to_string() const
    {
        std::string str = "[";
        for (size_t i = 0; i < MAX_DIM; i++)
            str += std::to_string(shape_[i]) + ", ";
        str += "]";
        return str;
    }

    std::string indices_to_string(const std::vector<size_t>& indices) const
    {
        std::string str = "[";
        for (size_t i = 0; i < indices.size(); i++)
            str += std::to_string(indices[i]) + ", ";
        str += "]";
        return str;
    }

public:
    static constexpr size_t MAX_DIM = 4;

private:
    size_t shape_[MAX_DIM];
    size_t strides_[MAX_DIM];
    size_t n_dims_;
    size_t offset_;
    Tensor<DataType>* parent;
    DataType* data_;
};
