#include <gtest/gtest.h>

#include "tensor.h"

TEST(TensorInitialization, DefaultConstructor)
{
    Tensor<int> tensor;
    EXPECT_EQ(tensor.shape()[0], 0);
    ASSERT_EQ(tensor.size(), 0);
}

TEST(TensorInitialization, ConstructorWithShape)
{
    auto tensor1 = Tensor<int>(nullptr, { 2, 3 });
    EXPECT_EQ(tensor1.size(), 6);
    EXPECT_EQ(tensor1.shape()[0], 2);
    EXPECT_EQ(tensor1.shape()[1], 3);

    auto tensor2 = Tensor<int>(nullptr, { 2, 3, 4 });
    EXPECT_EQ(tensor2.size(), 24);
    EXPECT_EQ(tensor2.shape()[0], 2);
    EXPECT_EQ(tensor2.shape()[1], 3);
    EXPECT_EQ(tensor2.shape()[2], 4);

    auto tensor3 = Tensor<int>(nullptr, { 2, 3, 4, 5 });
    EXPECT_EQ(tensor3.size(), 120);
    EXPECT_EQ(tensor3.shape()[0], 2);
    EXPECT_EQ(tensor3.shape()[1], 3);
    EXPECT_EQ(tensor3.shape()[2], 4);
    EXPECT_EQ(tensor3.shape()[3], 5);
}

TEST(TensorConstructorTest, InvalidShape) {
    std::initializer_list<size_t> shape = {1, 2, 3, 4, 5};
    EXPECT_THROW(Tensor<int>(nullptr, shape), std::invalid_argument);
}