#include <gtest/gtest.h>

#include "tensile/tensor.h"
#include "test_utils.h"

using std::make_tuple;
using std::pair;
using std::string;
using std::tuple;
using std::vector;
using Tensile::Tensor;

TEST(AddTensorTest, AddEmptyTensor)
{
    auto tensor1 = create_tensor({ 0, 0 });
    auto tensor2 = create_tensor({ 0, 0 });
    auto result = tensor1 + tensor2;
    ASSERT_EQ(result.n_dims(), 0);
    ASSERT_EQ(result.is_empty(), true);
}

TEST(AddTensorTest, AddSingleDimensionalTensor)
{
    auto tensor1 = create_tensor({ 5 });
    auto tensor2 = create_tensor({ 5 });

    auto result = tensor1 + tensor2;

    ASSERT_EQ(result.n_dims(), 1);
    ASSERT_EQ(result.size(), 5);
    ASSERT_EQ(result.shape()[0], 5);
    ASSERT_EQ(result.flat_string(), "[0, 2, 4, 6, 8, ]");
}

TEST(AddTensorTest, AddTwoDimensionalTensor)
{
    auto tensor1 = create_tensor({ 3, 3 });
    auto tensor2 = create_tensor({ 3, 3 });

    auto result = tensor1 + tensor2;

    ASSERT_EQ(result.n_dims(), 2);
    ASSERT_EQ(result.size(), 9);
    ASSERT_EQ(result.shape()[0], 3);
    ASSERT_EQ(result.shape()[1], 3);
    ASSERT_EQ(result.flat_string(), "[0, 2, 4, 6, 8, 10, 12, 14, 16, ]");
}

TEST(AddTensorTest, AddSingleDimensionalTensorWithDifferentSizes)
{
    auto tensor1 = create_tensor({ 3, 1 });
    auto tensor2 = create_tensor({ 3, 3 });

    auto result = tensor1 + tensor2;

    ASSERT_EQ(result.n_dims(), 2);
    ASSERT_EQ(result.size(), 9);

    ASSERT_EQ(result.shape()[0], 3);
    ASSERT_EQ(result.shape()[1], 3);

    ASSERT_EQ(result.flat_string(), "[0, 1, 2, 4, 5, 6, 8, 9, 10, ]");
}

TEST(AddTensorTest, AddTwoDimensionalTensorWithDifferentSizes)
{
    auto tensor1 = create_tensor({ 3, 1 });
    auto tensor2 = create_tensor({ 1, 3 });

    auto result = tensor1 + tensor2;

    ASSERT_EQ(result.n_dims(), 2);
    ASSERT_EQ(result.size(), 9);

    ASSERT_EQ(result.shape()[0], 3);
    ASSERT_EQ(result.shape()[1], 3);

    ASSERT_EQ(result.flat_string(), "[0, 1, 2, 1, 2, 3, 2, 3, 4, ]");
}