#include <gtest/gtest.h>
#include <tuple>
#include <utility>
#include <vector>

#include "tensor.h"

size_t flat_size(const std::vector<size_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

// Test fixture for tensor slicing tests. Takes in 4 parameters:
// 1. The shape of the tensor (std::vector<size_t>).
// 2. The indices to slice the tensor (std::vector<std::pair<size_t, size_t>).
// 3. The expected shape of the sliced tensor (std::vector<size_t>).
// 4. The expected values of the sliced tensor (string).
class TensorSlicingTest
    : public ::testing::TestWithParam<
          std::tuple<std::vector<size_t>, std::vector<std::pair<size_t, size_t>>, std::vector<size_t>, std::string>> {
protected:
    Tensor<int>* tensor;
    Tensor<int> sliced_tensor;
    std::vector<size_t> shape;
    std::vector<std::pair<size_t, size_t>> indices;
    std::vector<size_t> expected_shape;
    std::string expected_values;
    size_t len;
    int* data;

    void SetUp() override
    {
        shape = std::get<0>(GetParam());
        indices = std::get<1>(GetParam());
        expected_shape = std::get<2>(GetParam());
        expected_values = std::get<3>(GetParam());
        len = flat_size(shape);

        data = new int[len];

        for (int i = 0; i < len; i++)
            data[i] = i;

        tensor = new Tensor<int>(data, shape);

        sliced_tensor = tensor->operator[](indices);
    }

    void TearDown() override { delete tensor; }
};

TEST_P(TensorSlicingTest, Slicing)
{
    ASSERT_EQ(sliced_tensor.n_dims(), expected_shape.size());
    ASSERT_EQ(sliced_tensor.size(), flat_size(expected_shape));
    for (int i = 0; i < sliced_tensor.n_dims(); i++)
        ASSERT_EQ(sliced_tensor.shape()[i], expected_shape[i]);
    ASSERT_EQ(sliced_tensor.flat_string(), expected_values);
};

INSTANTIATE_TEST_SUITE_P(
    TensorSlicingTests, TensorSlicingTest,
    ::testing::Values(
        // clang-format off
        // One-dimensional tensor, one-dimensional slice
        std::make_tuple(std::vector<size_t>{3}, std::vector<std::pair<size_t, size_t>>{{0, 2}}, std::vector<size_t>{2}, "[0, 1, ]"),
        std::make_tuple(std::vector<size_t>{3}, std::vector<std::pair<size_t, size_t>>{{0, 3}}, std::vector<size_t>{3}, "[0, 1, 2, ]"),
        std::make_tuple(std::vector<size_t>{3}, std::vector<std::pair<size_t, size_t>>{{1, 3}}, std::vector<size_t>{2}, "[1, 2, ]"),
        std::make_tuple(std::vector<size_t>{3}, std::vector<std::pair<size_t, size_t>>{{1, 2}}, std::vector<size_t>{1}, "[1, ]"),
        std::make_tuple(std::vector<size_t>{9}, std::vector<std::pair<size_t, size_t>>{{0, 9}}, std::vector<size_t>{9}, "[0, 1, 2, 3, 4, 5, 6, 7, 8, ]"),
        std::make_tuple(std::vector<size_t>{9}, std::vector<std::pair<size_t, size_t>>{{0, 3}}, std::vector<size_t>{3}, "[0, 1, 2, ]"),
        std::make_tuple(std::vector<size_t>{9}, std::vector<std::pair<size_t, size_t>>{{1, 9}}, std::vector<size_t>{8}, "[1, 2, 3, 4, 5, 6, 7, 8, ]"),

        // Two-dimensional tensor, one-dimensional slice
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 3}}, std::vector<size_t>{2, 3}, "[0, 1, 2, 3, 4, 5, ]"),
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 3}, {0, 3}}, std::vector<size_t>{3, 3}, "[0, 1, 2, 3, 4, 5, 6, 7, 8, ]"),

        // Two-dimensional tensor, two-dimensional slice
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}}, std::vector<size_t>{2, 2}, "[0, 1, 3, 4, ]"),
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 3}, {0, 3}}, std::vector<size_t>{3, 3}, "[0, 1, 2, 3, 4, 5, 6, 7, 8, ]"),
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{1, 3}, {1, 3}}, std::vector<size_t>{2, 2}, "[4, 5, 7, 8, ]"),
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{1, 2}, {1, 2}}, std::vector<size_t>{1, 1}, "[4, ]"),
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 3}, {1, 3}}, std::vector<size_t>{3, 2}, "[1, 2, 4, 5, 7, 8, ]"),
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 3}, {0, 2}}, std::vector<size_t>{3, 2}, "[0, 1, 3, 4, 6, 7, ]"),
        std::make_tuple(std::vector<size_t>{3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 2}, {1, 3}}, std::vector<size_t>{2, 2}, "[1, 2, 4, 5, ]"),

        // Three-dimensional tensor, three-dimensional slice
        std::make_tuple(std::vector<size_t>{3, 3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}, {0, 2}}, std::vector<size_t>{2, 2, 2}, "[0, 1, 3, 4, 9, 10, 12, 13, ]"),
        std::make_tuple(std::vector<size_t>{3, 3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 3}, {0, 3}, {0, 3}}, std::vector<size_t>{3, 3, 3}, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, ]"),
        std::make_tuple(std::vector<size_t>{3, 3, 3}, std::vector<std::pair<size_t, size_t>>{{1, 3}, {1, 3}, {1, 3}}, std::vector<size_t>{2, 2, 2}, "[13, 14, 16, 17, 22, 23, 25, 26, ]"),
        std::make_tuple(std::vector<size_t>{3, 3, 3}, std::vector<std::pair<size_t, size_t>>{{1, 2}, {1, 2}, {1, 2}}, std::vector<size_t>{1, 1, 1}, "[13, ]"),
        std::make_tuple(std::vector<size_t>{3, 3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 3}, {1, 3}, {1, 3}}, std::vector<size_t>{3, 2, 2}, "[4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26, ]"),
        std::make_tuple(std::vector<size_t>{3, 3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 3}, {0, 2}, {0, 2}}, std::vector<size_t>{3, 2, 2}, "[0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22, ]"),
        std::make_tuple(std::vector<size_t>{3, 3, 3}, std::vector<std::pair<size_t, size_t>>{{0, 2}, {1, 3}, {1, 3}}, std::vector<size_t>{2, 2, 2}, "[4, 5, 7, 8, 13, 14, 16, 17, ]")
        // clang-format on
        ));