#include <gtest/gtest.h>

#include "tensor.h"
#include "test_utils.h"

using std::make_tuple;
using std::pair;
using std::string;
using std::tuple;
using std::vector;
using Tensile::Tensor;

TEST(TensorUnaryOpsTest, UnaryMinus)
{
    auto tensor = create_tensor({ 3, 3 });
    auto result = -tensor;

    ASSERT_EQ(result.n_dims(), 2);
    ASSERT_EQ(result.size(), 9);
    ASSERT_EQ(result.shape()[0], 3);
    ASSERT_EQ(result.shape()[1], 3);

    ASSERT_EQ(result.flat_string(), "[0, -1, -2, -3, -4, -5, -6, -7, -8, ]");
}

TEST(TensorUnaryOpsTest, UnaryPlus)
{
    auto tensor = create_tensor({ 3, 3 });
    auto result = +tensor;

    ASSERT_EQ(result.n_dims(), 2);
    ASSERT_EQ(result.size(), 9);
    ASSERT_EQ(result.shape()[0], 3);
    ASSERT_EQ(result.shape()[1], 3);

    ASSERT_EQ(result.flat_string(), "[0, 1, 2, 3, 4, 5, 6, 7, 8, ]");
}
