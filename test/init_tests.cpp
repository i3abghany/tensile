#include <gtest/gtest.h>

#include "tensor.h"

TEST(TensorInitialization, DefaultConstructor)
{
    Tensor<int> tensor;
    ASSERT_EQ(tensor.size(), 0);
}

TEST(TensorInitialization, ConstructorWithShape)
{
    auto tensor1 = Tensor<int>(nullptr, { 2, 3 });
    EXPECT_EQ(tensor1.size(), 6);

    auto tensor2 = Tensor<int>(nullptr, { 2, 3, 4 });
    EXPECT_EQ(tensor2.size(), 24);

    auto tensor3 = Tensor<int>(nullptr, { 2, 3, 4, 5 });
    EXPECT_EQ(tensor3.size(), 120);

    // This tensor's shape has more than 4 dimensions, which is not supported
    EXPECT_THROW(Tensor<int>(nullptr, { 2, 3, 4, 5, 6 }), std::invalid_argument);
}