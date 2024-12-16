#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <utility>

#include "index_parser.h"
#include "logger.h"

namespace Tensile {

template <typename D, typename S>
concept CompatibleTypes
    = (std::floating_point<D> && std::floating_point<S>) || (std::integral<D> && std::is_same_v<D, S>);

template <typename D>
concept TensorType = std::integral<D> || std::floating_point<D>;

static constexpr size_t MAX_DIM = 4;

template <typename DataType>
    requires TensorType<DataType>
class Tensor {
public:
    Tensor()
        : n_dims_(1)
        , offset_(0)
        , parent(nullptr)
        , data_(nullptr)
    {
        std::fill(shape_.begin(), shape_.end(), 0);
        std::fill(strides_.begin(), strides_.end(), 0);
    }

    Tensor(DataType* data, const std::vector<size_t>& pshape)
        : n_dims_(0)
        , offset_(0)
        , parent(nullptr)
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

    Tensor(DataType* data, std::array<size_t, MAX_DIM> shape)
        : Tensor(data, std::vector<size_t>(shape.begin(), shape.end()))
    {
    }

    Tensor(const Tensor<DataType>& other)
        : n_dims_(other.n_dims_)
        , offset_(other.offset_)
        , parent(other.parent)
        , data_(other.data_)
    {
        std::copy(other.shape_.begin(), other.shape_.end(), shape_.begin());
        std::copy(other.strides_.begin(), other.strides_.end(), strides_.begin());
    }

    Tensor& operator=(const Tensor<DataType>& other)
    {
        if (this == &other)
            return *this;

        n_dims_ = other.n_dims_;
        offset_ = other.offset_;
        parent = other.parent;
        data_ = other.data_;

        std::copy(other.shape_.begin(), other.shape_.end(), shape_.begin());
        std::copy(other.strides_.begin(), other.strides_.end(), strides_.begin());

        return *this;
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
        auto parsed_indices = parse_indices(indices);
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
        for (int i = (int)n_dims_ - 2; i >= 0; i--)
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

    template <typename D, typename S> static bool shape_compat(const Tensor<D>& a, const Tensor<S>& b)
    {
        auto a_dims = a.n_dims(), b_dims = b.n_dims();

        if (a_dims == b_dims) {
            for (size_t i = 0; i < a_dims; i++) {
                if (a.shape()[i] != b.shape()[i] && a.shape()[i] != 1 && b.shape()[i] != 1)
                    return false;
            }
            return true;
        }
        // FIXME: Implement the case where one of the tensors has fewer dimensions
        return false;
    }

    template <typename OtherDataType>
        requires CompatibleTypes<DataType, OtherDataType>
    auto operator+(const Tensor<OtherDataType>& other) -> Tensor<decltype(DataType() + OtherDataType())>
    {
        using ResultDataType = decltype(DataType() + OtherDataType());
        std::function<ResultDataType(DataType, OtherDataType)> op
            = [](DataType a, OtherDataType b) -> ResultDataType { return a + b; };
        return binary_broadcastable_elementwise_op(other, op);
    }

    template <typename OtherDataType>
        requires CompatibleTypes<DataType, OtherDataType>
    auto operator-(const Tensor<OtherDataType>& other) -> Tensor<decltype(DataType() - OtherDataType())>
    {
        using ResultDataType = decltype(DataType() - OtherDataType());
        std::function<ResultDataType(DataType, OtherDataType)> op
            = [](DataType a, OtherDataType b) -> ResultDataType { return a - b; };
        return binary_broadcastable_elementwise_op(other, op);
    }

    Tensor<DataType> operator+() const
    {
        return copy(); // identity operation
    }

    Tensor<DataType> operator-() const
    {
        std::function<DataType(DataType)> op = [](DataType a) -> DataType { return -a; };
        return unary_op(op);
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
    template <typename OtherDataType>
        requires CompatibleTypes<DataType, OtherDataType>
    auto binary_broadcastable_elementwise_op(
        const Tensor<OtherDataType>& other,
        std::function<decltype(DataType() + OtherDataType())(DataType, OtherDataType)> op)
    {
        if (!shape_compat(*this, other))
            throw std::invalid_argument("Incompatible shapes for element-wise operation");

        std::array<size_t, MAX_DIM> shape;
        for (size_t i = 0; i < n_dims_; i++)
            shape[i] = std::max(shape_[i], other.shape()[i]);

        for (size_t i = n_dims_; i < MAX_DIM; i++)
            shape[i] = 0;

        using ResultType = decltype(DataType() + OtherDataType());
        auto flat_size = std::accumulate(shape.begin(), shape.begin() + n_dims_, 1, std::multiplies<size_t>());
        auto* result_data = new ResultType[flat_size];

        std::array<size_t, MAX_DIM> iter;
        for (size_t i = 0; i < MAX_DIM; i++)
            iter[i] = shape[i] == 0 ? 1 : shape[i];

        Tensor<ResultType> result(result_data, shape);

        for (size_t i = 0; i < iter[0]; i++) {
            for (size_t j = 0; j < iter[1]; j++) {
                for (size_t k = 0; k < iter[2]; k++) {
                    for (size_t l = 0; l < iter[3]; l++) {
                        size_t aidx = broadcasted_flat_index({ i, j, k, l });
                        size_t bidx = other.broadcasted_flat_index({ i, j, k, l });
                        size_t flat_idx = result.multi_indices_to_flat({ i, j, k, l });
                        result_data[flat_idx] = op(data_[aidx], other.data_[bidx]);
                    }
                }
            }
        }
        return result;
    }

    auto unary_op(std::function<DataType(DataType)> op) -> Tensor<DataType>
    {
        auto new_tensor = copy();
        for (size_t i = 0; i < new_tensor.size(); i++)
            new_tensor.data_[i] = op(new_tensor.data_[i]);
        return new_tensor;
    }

private:
    size_t multi_indices_to_flat(const std::vector<size_t>& indices) const
    {
        size_t flat_idx = 0;
        for (int i = (int)n_dims_ - 1; i >= 0; i--) {
            if (indices[i] >= shape_[i] && (shape_[i] != 0 || indices[i] != 0)) {
                throw std::out_of_range("Index out of bounds");
            }
            flat_idx += indices[i] * strides_[i];
        }
        return flat_idx + offset_;
    }

    size_t broadcasted_flat_index(const std::vector<size_t>& indices) const
    {
        std::vector<size_t> new_indices;

        for (size_t i = 0; i < n_dims_; i++) {
            new_indices.push_back(indices[i] >= shape_[i] && shape_[i] == 1 ? 0 : indices[i]);
        }

        return multi_indices_to_flat(new_indices);
    }

    std::string shape_to_string() const
    {
        std::string str = "[";
        for (size_t i = 0; i < MAX_DIM; i++)
            str += std::to_string(shape_[i]) + ", ";
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