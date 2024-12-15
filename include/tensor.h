#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <utility>

#include "index_parser.h"
#include "logger.h"

namespace Tensile {

static constexpr size_t MAX_DIM = 4;

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
        std::fill(shape_.begin(), shape_.end(), 0);
        std::fill(strides_.begin(), strides_.end(), 0);
    }

    Tensor(DataType* data, const std::vector<size_t>& pshape)
        : parent(nullptr)
        , n_dims_(0)
        , offset_(0)
        , data_(data)
    {
        if (pshape.size() > MAX_DIM)
            throw std::invalid_argument("Tensor shape cannot have more than 4 dimensions");

        n_dims_ = pshape.size();
        for (int i = pshape.size() - 1; i >= 0; i--)
            if (pshape[i] == 0)
                n_dims_--;

        std::copy(pshape.begin(), pshape.end(), shape_.begin());
        for (size_t i = pshape.size(); i < MAX_DIM; i++)
            shape_[i] = 0;

        init_strides();
    }

    Tensor(DataType* data, size_t shape[MAX_DIM])
        : Tensor(data, std::vector<size_t>(shape, shape + MAX_DIM))
    {
    }

    // Shallow copy constructor
    Tensor(const Tensor<DataType>& other)
        : parent(other.parent)
        , n_dims_(other.n_dims_)
        , offset_(other.offset_)
        , data_(other.data_)
    {
        std::copy(other.shape_.begin(), other.shape_.end(), shape_.begin());
        std::copy(other.strides_.begin(), other.strides_.end(), strides_.begin());
    }

    Tensor copy() const
    {
        Tensor<DataType> new_tensor;
        new_tensor.data_ = new DataType[size()];
        new_tensor.n_dims_ = n_dims_;
        new_tensor.offset_ = 0;
        new_tensor.parent = nullptr;
        std::copy(shape_.begin(), shape_.end(), new_tensor.shape_.begin());
        std::copy(strides_.begin(), strides_.end(), new_tensor.strides_.begin());
        std::vector<size_t> iter(shape_.size(), 0);
        std::transform(iter.begin(), iter.end(), [](size_t d) { return d == 0 ? 1 : d; });
        for (size_t i = 0; i < iter[0]; i++) {
            for (size_t j = 0; j < iter[1]; j++) {
                for (size_t k = 0; k < iter[2]; k++) {
                    for (size_t l = 0; l < iter[3]; l++) {
                        size_t flat_idx = multi_indices_to_flat({ i, j, k, l });
                        new_tensor.data_[flat_idx] = data_[flat_idx];
                    }
                }
            }
        }
    }

    size_t size() const
    {
        size_t size = 1;
        for (size_t i = 0; i < n_dims_; i++)
            size *= shape_[i];
        return size;
    }

    Tensor<DataType> operator[](const std::string& indices)
    {
        std::vector<std::pair<size_t, size_t>> parsed_indices = parse_indices(indices);
        return operator[](parsed_indices);
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

    void expand_dims(size_t axis)
    {
        if (axis > n_dims_)
            throw std::invalid_argument("Axis out of bounds");

        if (n_dims_ == MAX_DIM)
            throw std::invalid_argument("Cannot expand dimensions");

        for (size_t i = n_dims_; i > axis; i--) {
            shape_[i] = shape_[i - 1];
            strides_[i] = strides_[i - 1];
        }

        shape_[axis] = 1;
        strides_[axis] = 0;
        n_dims_++;
    }

    void squeeze(size_t axis)
    {
        if (axis >= n_dims_)
            throw std::invalid_argument("Axis out of bounds");

        for (size_t i = axis; i < n_dims_ - 1; i++) {
            shape_[i] = shape_[i + 1];
            strides_[i] = strides_[i + 1];
        }

        shape_[n_dims_ - 1] = 0;
        strides_[n_dims_ - 1] = 0;
        n_dims_--;
    }

    bool is_empty() const { return n_dims_ == 0; }

    const std::array<size_t, MAX_DIM> shape() const { return shape_; }

    size_t n_dims() const { return n_dims_; }

    ~Tensor()
    {
        if (parent == nullptr)
            delete[] data_;
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

private:
    std::array<size_t, MAX_DIM> shape_;
    std::array<size_t, MAX_DIM> strides_;
    size_t n_dims_;
    size_t offset_;
    Tensor<DataType>* parent;
    DataType* data_;
};

}