#include <gtest/gtest.h>

#include "tensor.h"
#include "test_utils.h"

using std::make_tuple;
using std::vector;
using std::tuple;
using Tensile::Tensor;

// Test fixture for tensor expand dims tests. Takes in 3 parameters:
// 1. The shape of the tensor (std::vector<size_t>).
// 2. The axis to expand the tensor (size_t).
// 3. The expected shape of the expanded tensor (std::vector<size_t>).

class ExpandDimsTest : public ::testing::TestWithParam<tuple<vector<size_t>, size_t, vector<size_t>>>
{
protected:
    Tensor<int>* tensor;
    vector<size_t> shape;
    size_t axis;
    vector<size_t> expected_shape;
    size_t len;
    int* data;

    void SetUp() override
    {
        shape = std::get<0>(GetParam());
        axis = std::get<1>(GetParam());
        expected_shape = std::get<2>(GetParam());
        len = flat_size(shape);

        data = new int[len];
        tensor = new Tensor<int>(data, shape);
    }

    void TearDown() override { delete tensor; }
};

TEST_P(ExpandDimsTest, ExpandDims)
{
    tensor->expand_dims(axis);
    ASSERT_EQ(tensor->n_dims(), expected_shape.size());
    for (size_t i = 0; i < tensor->n_dims(); i++)
        ASSERT_EQ(tensor->shape()[i], expected_shape[i]);
};

INSTANTIATE_TEST_SUITE_P(
    TensorExpandDimsTests, ExpandDimsTest,
    ::testing::Values(
        // One-dimensional tensor, expand to 2 dimensions
        make_tuple(vector<size_t>{3}, 0, vector<size_t>{1, 3}),
        make_tuple(vector<size_t>{3}, 1, vector<size_t>{3, 1}),
        // Two-dimensional tensor, expand to 3 dimensions
        make_tuple(vector<size_t>{3, 3}, 0, vector<size_t>{1, 3, 3}),
        make_tuple(vector<size_t>{3, 3}, 1, vector<size_t>{3, 1, 3}),
        make_tuple(vector<size_t>{3, 3}, 2, vector<size_t>{3, 3, 1}),
        // Three-dimensional tensor, expand to 4 dimensions
        make_tuple(vector<size_t>{3, 3, 3}, 0, vector<size_t>{1, 3, 3, 3}),
        make_tuple(vector<size_t>{3, 3, 3}, 1, vector<size_t>{3, 1, 3, 3}),
        make_tuple(vector<size_t>{3, 3, 3}, 2, vector<size_t>{3, 3, 1, 3}),
        make_tuple(vector<size_t>{3, 3, 3}, 3, vector<size_t>{3, 3, 3, 1})
    ));

class ExpandDimsThrowsTest : public ::testing::TestWithParam<tuple<vector<size_t>, size_t>>
{
protected:
    Tensor<int>* tensor;
    vector<size_t> shape;
    size_t axis;
    size_t len;
    int* data;

    void SetUp() override
    {
        shape = std::get<0>(GetParam());
        axis = std::get<1>(GetParam());
        len = flat_size(shape);

        data = new int[len];
        tensor = new Tensor<int>(data, shape);
    }

    void TearDown() override { delete tensor; }
};

TEST_P(ExpandDimsThrowsTest, ExpandDimsThrows)
{
    EXPECT_THROW(tensor->expand_dims(axis), std::invalid_argument);
};

INSTANTIATE_TEST_SUITE_P(
    TensorExpandDimsThrowsTests, ExpandDimsThrowsTest,
    ::testing::Values(
        make_tuple(vector<size_t>{3, 3, 3, 3}, 0),
        make_tuple(vector<size_t>{3, 3, 3, 3}, 1),
        make_tuple(vector<size_t>{3, 3, 3, 3}, 2),
        make_tuple(vector<size_t>{3, 3, 3, 3}, 3),
        make_tuple(vector<size_t>{3, 3, 3, 3}, 4),
        make_tuple(vector<size_t>{3, 3, 3, 3}, 5),
        make_tuple(vector<size_t>{3, 3, 3}, 6),
        make_tuple(vector<size_t>{3, 3}, 4)
    ));


// Test fixture for tensor squeeze tests. Takes in 3 parameters:
// 1. The shape of the tensor (std::vector<size_t>).
// 2. The axis to squeeze the tensor (size_t).
// 3. The expected shape of the squeezed tensor (std::vector<size_t>).

class SqueezeTest : public ::testing::TestWithParam<tuple<vector<size_t>, size_t, vector<size_t>>>
{
protected:
    Tensor<int>* tensor;
    vector<size_t> shape;
    size_t axis;
    vector<size_t> expected_shape;
    size_t len;
    int* data;

    void SetUp() override
    {
        shape = std::get<0>(GetParam());
        axis = std::get<1>(GetParam());
        expected_shape = std::get<2>(GetParam());
        len = flat_size(shape);

        data = new int[len];
        tensor = new Tensor<int>(data, shape);
    }

    void TearDown() override { delete tensor; }
};

TEST_P(SqueezeTest, Squeeze)
{
    tensor->squeeze(axis);
    ASSERT_EQ(tensor->n_dims(), expected_shape.size());
    for (size_t i = 0; i < tensor->n_dims(); i++)
        ASSERT_EQ(tensor->shape()[i], expected_shape[i]);
};

INSTANTIATE_TEST_SUITE_P(
    TensorSqueezeTests, SqueezeTest,
    ::testing::Values(
        make_tuple(vector<size_t>{1}, 0, vector<size_t>{}),

        make_tuple(vector<size_t>{1, 3}, 0, vector<size_t>{3}),
        make_tuple(vector<size_t>{3, 1}, 1, vector<size_t>{3}),

        make_tuple(vector<size_t>{1, 3, 1}, 0, vector<size_t>{3, 1}),
        make_tuple(vector<size_t>{3, 1, 1}, 2, vector<size_t>{3, 1}),
        make_tuple(vector<size_t>{1, 1, 3}, 1, vector<size_t>{1, 3}),

        make_tuple(vector<size_t>{1, 3, 1, 1}, 0, vector<size_t>{3, 1, 1}),
        make_tuple(vector<size_t>{1, 3, 1, 1}, 2, vector<size_t>{1, 3, 1}),
        make_tuple(vector<size_t>{1, 3, 1, 1}, 3, vector<size_t>{1, 3, 1}),
        make_tuple(vector<size_t>{3, 1, 1, 1}, 1, vector<size_t>{3, 1, 1}),
        make_tuple(vector<size_t>{3, 1, 1, 1}, 2, vector<size_t>{3, 1, 1}),
        make_tuple(vector<size_t>{3, 1, 1, 1}, 3, vector<size_t>{3, 1, 1})
    ));