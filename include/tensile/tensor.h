#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <numeric>
#include <utility>

#include "enumerate.h"
#include "index_parser.h"
#include "logger.h"
#include "unimpl.h"

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
        : n_dims_(0)
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

        n_dims_ = get_n_dims_from_shape(pshape);
        std::copy(pshape.begin(), pshape.end(), shape_.begin());
        init_strides();
    }

    Tensor(DataType* data, size_t shape[MAX_DIM])
        : Tensor(data, std::vector<size_t>(shape, shape + MAX_DIM))
    {
    }

    template <size_t N>
    Tensor(DataType* data, std::array<size_t, N> shape)
        : Tensor(data, std::vector<size_t>(shape.begin(), shape.end()))
    {
    }

    Tensor(const Tensor<DataType>& other)
        : n_dims_(other.n_dims_)
        , offset_(other.offset_)
        , parent(!other.parent ? &const_cast<Tensor<DataType>&>(other) : other.parent)
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
        parent = !other.parent ? &const_cast<Tensor<DataType>&>(other) : other.parent;
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

        auto iter = get_iter_shape(shape_);
        ENUMERATE(iter, i, j, k, l)
        {
            size_t flat_idx = multi_indices_to_flat({ i, j, k, l });
            new_tensor.data_[flat_idx] = data_[flat_idx];
        }

        return new_tensor;
    }

    [[nodiscard]] size_t size() const
    {
        size_t size = 1;
        for (size_t i = 0; i < std::max((size_t)1, n_dims_); i++)
            size *= shape_[i];
        return size;
    }

    bool operator==(const Tensor<DataType>& other) const
    {
        if (n_dims_ != other.n_dims_)
            return false;

        for (size_t i = 0; i < n_dims_; i++)
            if (shape_[i] != other.shape_[i])
                return false;

        auto iter = get_iter_shape(shape_);
        ENUMERATE(iter, i, j, k, l)
        {
            size_t flat_idx = multi_indices_to_flat({ i, j, k, l });
            if (data_[flat_idx] != other.data_[flat_idx])
                return false;
        }

        return true;
    }

    Tensor<DataType> operator[](const std::string& indices) const
    {
        auto parsed_indices = parse_indices(indices);
        return operator[](parsed_indices);
    }

    DataType item() const
    {
        if (n_dims_ != 1 || shape_[0] != 1)
            throw std::invalid_argument("Can only call item() on a 1D tensor with one element");
        return data_[offset_];
    }

    DataType& operator[](const std::vector<size_t>& indices)
    {
        if (indices.size() != n_dims_)
            throw std::invalid_argument("Number of indices does not match the number of dimensions");

        size_t flat_idx = multi_indices_to_flat(indices);
        return data_[flat_idx];
    }

    DataType operator[](const std::vector<size_t>& indices) const
    {
        if (indices.size() != n_dims_)
            throw std::invalid_argument("Number of indices does not match the number of dimensions");

        size_t flat_idx = multi_indices_to_flat(indices);
        return data_[flat_idx];
    }

    const DataType* address_of(const std::vector<size_t>& indices) const
    {
        if (indices.size() != n_dims_)
            throw std::invalid_argument("Number of indices does not match the number of dimensions");

        size_t flat_idx = multi_indices_to_flat(indices);
        return &data_[flat_idx];
    }

    DataType& item_at(const std::vector<size_t>& indices) { return operator[](indices); }

    DataType item_at(const std::vector<size_t>& indices) const { return operator[](indices); }

    Tensor<DataType> operator[](const std::vector<std::pair<size_t, size_t>>& indices) const
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
        std::fill(strides_.begin(), strides_.end(), 0);
        if (n_dims_ == 0)
            return;

        strides_[n_dims_ - 1] = 1;
        for (int i = (int)n_dims_ - 2; i >= 0; i--)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }

    [[nodiscard]] std::string flat_string() const
    {
        auto iter = get_iter_shape(shape_);
        std::string str = "[";

        ENUMERATE(iter, i, j, k, l)
        {
            auto flat_idx = multi_indices_to_flat({ i, j, k, l });
            str += std::to_string(data_[flat_idx]) + ", ";
        }

        str += "]";
        return str;
    }

    [[nodiscard]] std::string to_string() const
    {
        std::string res = "Tensor(";
        res += to_string_rec() + ", shape=" + shape_to_string() + ")";
        return res;
    }

    Tensor<DataType> sum(size_t axis, bool keepdims) const
    {
        if (axis >= n_dims_)
            throw std::invalid_argument("Axis out of bounds");

        std::vector<size_t> new_shape {};
        for (size_t i = 0; i < n_dims_; i++) {
            if (i == axis)
                continue;
            new_shape.push_back(shape_[i]);
        }

        auto result = zeros(new_shape);

        auto iter = get_iter_shape(shape_);
        ENUMERATE(iter, i, j, k, l)
        {
            std::vector<size_t> indices = { i, j, k, l };
            indices.erase(indices.begin() + axis);
            size_t new_idx = result.broadcasted_flat_index(indices);
            size_t original_idx = broadcasted_flat_index({ i, j, k, l });
            result.data_[new_idx] += data_[original_idx];
        }

        if (keepdims)
            result.expand_dims(axis);

        return result;
    }

    Tensor<DataType> transpose() const
    {
        if (n_dims_ != 2)
            throw std::invalid_argument("Transpose is only implemented for 2D tensors");

        Tensor<DataType> result(*this);

        auto new_shape = shape_;
        std::swap(new_shape[0], new_shape[1]);

        auto new_strides = strides_;
        std::swap(new_strides[0], new_strides[1]);

        std::copy(new_shape.begin(), new_shape.end(), result.shape_.begin());
        std::copy(new_strides.begin(), new_strides.end(), result.strides_.begin());

        return result;
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

    template <typename D, typename S> static bool matmul_compat(const Tensor<D>& a, const Tensor<S>& b)
    {
        if (a.n_dims() == 2 && b.n_dims() == 2) {
            return a.shape()[1] == b.shape()[0];
        } else if (a.n_dims() == 3 && b.n_dims() == 3) {
            return a.shape()[2] == b.shape()[1]
                && (a.shape()[0] == b.shape()[0] || a.shape()[0] == 1 || b.shape()[0] == 1);
        }

        UNIMPLEMENTED("Matmul only implemented for tensors of the same number of dimensions");
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

        // FIXME: broadcasting not implemented for tensors with different number
        // of dimensions. For now we just return false
        return false;
    }

    template <typename OtherDataType>
    requires CompatibleTypes<DataType, OtherDataType>
    auto operator+(const Tensor<OtherDataType>& other) -> Tensor<decltype(DataType() + OtherDataType())> const
    {
        using ResultDataType = decltype(DataType() + OtherDataType());
        std::function<ResultDataType(DataType, OtherDataType)> op
            = [](DataType a, OtherDataType b) -> ResultDataType { return a + b; };
        return binary_broadcastable_elementwise_op(other, op);
    }

    template <typename OtherDataType>
    requires CompatibleTypes<DataType, OtherDataType>
    auto operator-(const Tensor<OtherDataType>& other) -> Tensor<decltype(DataType() - OtherDataType())> const
    {
        using ResultDataType = decltype(DataType() - OtherDataType());
        std::function<ResultDataType(DataType, OtherDataType)> op
            = [](DataType a, OtherDataType b) -> ResultDataType { return a - b; };
        return binary_broadcastable_elementwise_op(other, op);
    }

    template <typename OtherDataType>
    requires CompatibleTypes<DataType, OtherDataType>
    auto elementwise_mul(const Tensor<OtherDataType>& other) -> Tensor<decltype(DataType() * OtherDataType())> const
    {
        using ResultDataType = decltype(DataType() * OtherDataType());
        std::function<ResultDataType(DataType, OtherDataType)> op
            = [](DataType a, OtherDataType b) -> ResultDataType { return a * b; };
        return binary_broadcastable_elementwise_op(other, op);
    }

    template <typename OtherDataType>
    requires std::integral<OtherDataType>
    auto pow(OtherDataType exponent) -> Tensor<DataType> const
    {
        if (exponent == 0) {
            return ones({ shape_.begin(), shape_.begin() + n_dims_ });
        }

        if (exponent == 1) {
            return copy();
        }

        if (exponent % 2 == 0) {
            auto half_pow = pow(exponent / 2);
            return half_pow.elementwise_mul(half_pow);
        } else {
            return elementwise_mul(pow(exponent - 1));
        }
    }

    template <typename OtherDataType>
    requires CompatibleTypes<DataType, OtherDataType>
    auto operator*(const Tensor<OtherDataType>& other) -> Tensor<decltype(DataType() * OtherDataType())> const
    {
        if (!matmul_compat(*this, other))
            throw std::invalid_argument("Incompatible shapes for matrix multiplication");

        if (n_dims() == 2) {
            auto otherT = other.transpose();
            return matmul2d_transpose_other(otherT);
        }
        if (n_dims() == 3)
            return matmul3d(other);

        UNIMPLEMENTED("Matmul is not implemented for tensors with different number of dimensions");
    }

    template <typename OtherDataType>
    requires CompatibleTypes<DataType, OtherDataType>
    auto operator*(OtherDataType scalar) -> Tensor<decltype(DataType() * scalar)> const
    {
        std::function<decltype(DataType() * OtherDataType())(DataType)> op
            = [scalar](DataType a) -> decltype(DataType() * scalar) { return a * scalar; };
        return unary_op(op);
    }

    template <typename OtherDataType>
    requires CompatibleTypes<DataType, OtherDataType>
    auto operator+(OtherDataType scalar) -> Tensor<decltype(DataType() * scalar)> const
    {
        std::function<decltype(DataType() * OtherDataType())(DataType)> op
            = [scalar](DataType a) -> decltype(DataType() * scalar) { return a + scalar; };
        return unary_op(op);
    }

    Tensor<DataType> reciprocal() const
    {
        std::function<DataType(DataType)> op = [](DataType a) -> DataType { return 1 / a; };
        return unary_op(op);
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

    Tensor<DataType> exp() const
    {
        std::function<DataType(DataType)> op = [](DataType a) -> DataType { return std::exp(a); };
        return unary_op(op);
    }

    [[nodiscard]] bool is_empty() const { return n_dims_ == 0; }

    [[nodiscard]] std::array<size_t, MAX_DIM> shape() const { return shape_; }

    [[nodiscard]] size_t n_dims() const { return n_dims_; }

    ~Tensor()
    {
        if (parent == nullptr)
            delete[] data_;
    }

public:
    static Tensor<DataType> ones(const std::vector<size_t>& shape) { return all_v(shape, 1); }

    static Tensor<DataType> zeros(const std::vector<size_t>& shape) { return all_v(shape, 0); }

    static Tensor<DataType> rand(const std::vector<size_t>& shape)
    {
        size_t n_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        auto* data = new DataType[n_elems];
        for (size_t i = 0; i < n_elems; i++) {
            data[i] = (DataType)std::rand() / RAND_MAX;
        }
        return Tensor(data, shape);
    }

private:
    static Tensor<DataType> all_v(const std::vector<size_t>& shape, DataType value)
    {
        size_t n_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        auto* data = new DataType[n_elems];
        std::fill(data, data + n_elems, value);
        return Tensor(data, shape);
    }

private:
    template <typename OtherDataType>
    requires std::integral<DataType> && std::integral<OtherDataType>
    auto matmul2d_transpose_other(const Tensor<OtherDataType>& other)
        -> Tensor<decltype(DataType() * OtherDataType())> const
    {
        assert(n_dims() == 2 && other.n_dims() == 2 && matmul_compat(*this, other.transpose()));

        size_t a = shape()[0], b = shape()[1];
        size_t d = other.shape()[0];

        using ResultType = decltype(DataType() * OtherDataType());
        auto* result_data = new ResultType[a * d];
        Tensor<ResultType> result(result_data, { a, d });

#pragma omp parallel for num_threads(8)
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < d; j++) {
                ResultType sum = 0;
                for (size_t k = 0; k < b; k++)
                    sum += item_at({ i, k }) * other.item_at({ j, k });

                result.item_at({ i, j }) = sum;
            }
        }

        return result;
    }

    template <typename OtherDataType>
    requires std::floating_point<OtherDataType> && std::floating_point<DataType>
    auto matmul2d_transpose_other(const Tensor<OtherDataType>& other)
        -> Tensor<decltype(DataType() * OtherDataType())> const
    {
        assert(n_dims() == 2 && other.n_dims() == 2 && matmul_compat(*this, other.transpose()));

        size_t a = shape()[0], b = shape()[1];
        size_t d = other.shape()[0];

        using ResultType = decltype(DataType() * OtherDataType());
        auto* result_data = new ResultType[a * d];
        Tensor<ResultType> result(result_data, { a, d });

#pragma omp parallel for num_threads(8)
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < d; j++) {
                ResultType sum = 0;
                size_t k = 0;
                __m256 vec_sum = _mm256_setzero_ps();
                for (; k + 7 < b; k += 8) {
                    __m256 a_vec = _mm256_loadu_ps(address_of({ i, k }));
                    __m256 b_vec = _mm256_loadu_ps(other.address_of({ j, k }));
                    vec_sum = _mm256_fmadd_ps(a_vec, b_vec, vec_sum);
                }

                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, vec_sum);

                for (size_t l = 0; l < 8; l++)
                    sum += tmp[l];

                for (; k < b; k++)
                    sum += item_at({ i, k }) * other.item_at({ j, k });

                result.item_at({ i, j }) = sum;
            }
        }

        return result;
    }

    template <typename OtherDataType>
    requires CompatibleTypes<DataType, OtherDataType>
    auto matmul3d(const Tensor<OtherDataType>& other) -> Tensor<decltype(DataType() * OtherDataType())> const
    {
        assert(n_dims() == 3 && other.n_dims() == 3 && matmul_compat(*this, other));

        size_t batch = std::max(shape()[0], other.shape()[0]);
        size_t a = shape()[1], b = shape()[2];
        size_t d = other.shape()[1], e = other.shape()[2];

        using ResultType = decltype(DataType() * OtherDataType());
        auto* result_data = new ResultType[batch * d * e];
        Tensor<ResultType> result(result_data, { batch, d, e });

        for (size_t bt = 0; bt < batch; bt++) {
            for (size_t i = 0; i < a; i++) {
                for (size_t j = 0; j < e; j++) {
                    ResultType sum = 0;
                    for (size_t k = 0; k < b; k++) {
                        sum += item_at({ std::min(shape()[0] - 1, bt), i, k })
                            * other.item_at({ std::min(other.shape()[0] - 1, bt), k, j });
                    }
                    result.item_at({ bt, i, j }) = sum;
                }
            }
        }

        return result;
    }

private:
    template <typename OtherDataType>
    requires CompatibleTypes<DataType, OtherDataType>
    auto binary_broadcastable_elementwise_op(
        const Tensor<OtherDataType>& other,
        std::function<decltype(DataType() + OtherDataType())(DataType, OtherDataType)> op) const
    {
        if (!shape_compat(*this, other))
            throw std::invalid_argument("Incompatible shapes for element-wise operation");

        auto shape = get_broadcasted_shape(other.shape());
        auto result_n_dims = get_n_dims_from_shape(shape);
        auto iter = get_iter_shape(shape);

        using ResultType = decltype(DataType() + OtherDataType());
        auto flat_size = std::accumulate(shape.begin(), shape.begin() + result_n_dims, 1, std::multiplies<>());
        auto* result_data = new ResultType[flat_size];

        Tensor<ResultType> result(result_data, shape);

        ENUMERATE(iter, i, j, k, l)
        {
            size_t aidx = broadcasted_flat_index({ i, j, k, l });
            size_t bidx = other.broadcasted_flat_index({ i, j, k, l });
            size_t flat_idx = result.multi_indices_to_flat({ i, j, k, l });
            result_data[flat_idx] = op(data_[aidx], other.data_[bidx]);
        }
        return result;
    }

    auto unary_op(std::function<DataType(DataType)> op) -> Tensor<DataType> const
    {
        auto new_tensor = copy();
        for (size_t i = 0; i < new_tensor.size(); i++)
            new_tensor.data_[i] = op(new_tensor.data_[i]);
        return new_tensor;
    }

private:
    template <typename Iterable> static size_t get_n_dims_from_shape(const Iterable& shape)
    {
        assert(shape.size() <= MAX_DIM);
        size_t n_dims = shape.size();
        for (int i = shape.size() - 1; i >= 0; i--)
            if (shape[i] == 0)
                n_dims--;
        return n_dims;
    }

    static std::array<size_t, MAX_DIM> get_iter_shape(const std::array<size_t, MAX_DIM>& s)
    {
        std::array<size_t, MAX_DIM> iter {};
        for (size_t i = 0; i < MAX_DIM; i++)
            iter[i] = s[i] == 0 ? 1 : s[i];
        return iter;
    }

    // FIXME: doesn't work when tensors have different number of dimensions
    [[nodiscard]] std::array<size_t, MAX_DIM> get_broadcasted_shape(const std::array<size_t, MAX_DIM>& s2) const
    {
        UNIMPLEMENTED_IF(get_n_dims_from_shape(s2) != n_dims_,
                         "Broadcasting not implemented for tensors with different number of dimensions");
        std::array<size_t, MAX_DIM> shape {};
        for (size_t i = 0; i < n_dims_; i++)
            shape[i] = std::max(shape_[i], s2[i]);

        return shape;
    }

    [[nodiscard]] size_t multi_indices_to_flat(const std::vector<size_t>& indices) const
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

    [[nodiscard]] size_t broadcasted_flat_index(const std::vector<size_t>& indices) const
    {
        std::vector<size_t> new_indices;

        for (size_t i = 0; i < n_dims_; i++) {
            new_indices.push_back(indices[i] >= shape_[i] && shape_[i] == 1 ? 0 : indices[i]);
        }

        return multi_indices_to_flat(new_indices);
    }

    [[nodiscard]] std::string to_string_rec(std::vector<size_t> dims = {}) const
    {
        if (n_dims_ == 0)
            return "[]";

        if (dims.size() == n_dims_)
            return "";

        std::string res = "[";
        for (size_t i = 0; i < shape_[dims.size()]; i++) {
            dims.push_back(i);
            auto delim = (i < shape_[dims.size() - 1] - 1) ? ", " : "";
            if (dims.size() == n_dims_) {
                res += std::to_string(data_[multi_indices_to_flat(dims)]) + delim;
            } else {
                res += "\n" + std::string(dims.size() * 2, ' ') + to_string_rec(dims) + delim;
            }
            dims.pop_back();
        }
        res += (dims.size() == n_dims_ - 1 ? "]" : "\n" + std::string(dims.size() * 2, ' ') + "]");
        return res;
    }

    [[nodiscard]] std::string shape_to_string() const
    {
        std::string str = "[";
        for (size_t i = 0; i < n_dims_; i++)
            str += std::to_string(shape_[i]) + (i < n_dims_ - 1 ? ", " : "");
        str += "]";
        return str;
    }

private:
    std::array<size_t, MAX_DIM> shape_ { 0 };
    std::array<size_t, MAX_DIM> strides_ { 0 };
    size_t n_dims_ { 0 };
    size_t offset_ { 0 };
    const Tensor<DataType>* parent { nullptr };
    DataType* data_ { nullptr };
};

}