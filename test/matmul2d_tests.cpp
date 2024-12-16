#include <gtest/gtest.h>

#include "tensor.h"
#include "test_utils.h"

using std::array;
using std::make_tuple;
using std::pair;
using std::string;
using std::tuple;
using std::vector;
using Tensile::Tensor;

TEST(Tensor2dMatmulTest, Matmul)
{
    auto t1 = create_tensor({ 2, 2 });
    auto t2 = create_tensor({ 2, 2 });

    const auto result = t1 * t2;

    ASSERT_EQ((result[{ 0, 0 }]), 2);
    ASSERT_EQ((result[{ 0, 1 }]), 3);
    ASSERT_EQ((result[{ 1, 0 }]), 6);
    ASSERT_EQ((result[{ 1, 1 }]), 11);
}

TEST(Tensor2dMatmulTest, SliceMatmul)
{
    auto t1 = create_tensor({ 4, 4 });
    auto t2 = create_tensor({ 4, 4 });

    auto s1 = t1["1:3, 1:3"]; // [[5, 6], [9, 10]]
    auto s2 = t2["0:2, 0:2"]; // [[0, 1], [4, 5]]

    const auto result = s1 * s2;

    ASSERT_EQ((result[{ 0, 0 }]), 24);
    ASSERT_EQ((result[{ 0, 1 }]), 35);
    ASSERT_EQ((result[{ 1, 0 }]), 40);
    ASSERT_EQ((result[{ 1, 1 }]), 59);
}