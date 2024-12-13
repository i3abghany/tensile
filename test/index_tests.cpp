#include <gtest/gtest.h>

#include "tensor.h"

TEST(TensorIndexing, OutOfBounds_ShouldThrow)
{
    Tensor<int> tensor(nullptr, { 2, 3, 4 });
    EXPECT_THROW((tensor[{ 0, 0, 4 }]), std::out_of_range);
    EXPECT_THROW((tensor[{ 0, 3, 0 }]), std::out_of_range);
    EXPECT_THROW((tensor[{ 2, 0, 0 }]), std::out_of_range);
}

TEST(TensorIndexing, WrongShape_ShouldThrow)
{
    Tensor<int> tensor(nullptr, { 2, 3, 4 });
    EXPECT_THROW((tensor[{ 0, 0 }]), std::invalid_argument);
    EXPECT_THROW((tensor[{ 0, 0, 0, 0 }]), std::invalid_argument);
    EXPECT_THROW((tensor[{ 0 }]), std::invalid_argument);
}

TEST(TensorIndexing, SimpleIndexing)
{
    int* data = new int[24];
    for (int i = 0; i < 24; i++)
        data[i] = i;

    Tensor<int> tensor(data, { 2, 3, 4 });
    auto* shape = tensor.shape();

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 4; k++) {
                size_t flat_idx = i * shape[1] * shape[2] + j * shape[2] + k;
                EXPECT_EQ((tensor[{ i, j, k }]), data[flat_idx]);
            }
        }
    }

    delete[] data;
}