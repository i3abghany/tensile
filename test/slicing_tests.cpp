#include <gtest/gtest.h>

#include "tensor.h"

TEST(TensorSlicing, ValueSlice)
{
    std::vector<size_t> shape = { 3, 3 };
    size_t size = shape[0] * shape[1];
    int* data = new int[size];
    for (int i = 0; i < size; i++)
        data[i] = i;

    Tensor<int> tensor(data, shape);

    auto sub_tensor = tensor[{
        { 0, 1 },
        { 0, 1 }
    }];
    auto* sub_shape = sub_tensor.shape();

    EXPECT_EQ(sub_shape[0], 1);
    EXPECT_EQ(sub_shape[1], 1);
    EXPECT_STREQ(sub_tensor.flat_string().c_str(), "[0, ]");

    auto sub_tensor2 = tensor[{
        { 1, 2 },
        { 1, 2 }
    }];
    auto* sub_shape2 = sub_tensor2.shape();

    EXPECT_EQ(sub_shape2[0], 1);
    EXPECT_EQ(sub_shape2[1], 1);
    EXPECT_STREQ(sub_tensor2.flat_string().c_str(), "[4, ]");

    auto sub_tensor3 = tensor[{
        { 2, 3 },
        { 2, 3 }
    }];
    auto* sub_shape3 = sub_tensor3.shape();

    EXPECT_EQ(sub_shape3[0], 1);
    EXPECT_EQ(sub_shape3[1], 1);
    EXPECT_STREQ(sub_tensor3.flat_string().c_str(), "[8, ]");
}

TEST(TensorSlicing, EmptySlice)
{
    std::vector<size_t> shape = { 3, 3 };
    size_t size = shape[0] * shape[1];
    int* data = new int[size];
    for (int i = 0; i < size; i++)
        data[i] = i;

    Tensor<int> tensor(data, shape);

    auto sub_tensor = tensor[{
        { 0, 0 },
        { 0, 0 }
    }];
    auto* sub_shape = sub_tensor.shape();

    EXPECT_EQ(sub_shape[0], 0);
    EXPECT_EQ(sub_shape[1], 0);
    EXPECT_STREQ(sub_tensor.flat_string().c_str(), "[]");

    auto sub_tensor2 = tensor[{
        { 1, 1 },
        { 1, 1 }
    }];
    auto* sub_shape2 = sub_tensor2.shape();

    EXPECT_EQ(sub_shape2[0], 0);
    EXPECT_EQ(sub_shape2[1], 0);
    EXPECT_STREQ(sub_tensor2.flat_string().c_str(), "[]");

    auto sub_tensor3 = tensor[{
        { 2, 2 },
        { 2, 2 }
    }];
    auto* sub_shape3 = sub_tensor3.shape();

    EXPECT_EQ(sub_shape3[0], 0);
    EXPECT_EQ(sub_shape3[1], 0);
    EXPECT_STREQ(sub_tensor3.flat_string().c_str(), "[]");
}

TEST(TensorSlicing, SimpleSlice)
{
    std::vector<size_t> shape = { 3, 3 };
    size_t size = shape[0] * shape[1];
    int* data = new int[size];
    for (int i = 0; i < size; i++)
        data[i] = i;

    Tensor<int> tensor(data, shape);

    auto sub_tensor = tensor[{
        { 0, 2 },
        { 0, 2 }
    }];
    auto* sub_shape = sub_tensor.shape();

    EXPECT_EQ(sub_shape[0], 2);
    EXPECT_EQ(sub_shape[1], 2);
    EXPECT_STREQ(sub_tensor.flat_string().c_str(), "[0, 1, 3, 4, ]");

    auto sub_tensor2 = tensor[{
        { 1, 3 },
        { 1, 3 }
    }];
    auto* sub_shape2 = sub_tensor2.shape();

    EXPECT_EQ(sub_shape2[0], 2);
    EXPECT_EQ(sub_shape2[1], 2);

    EXPECT_STREQ(sub_tensor2.flat_string().c_str(), "[4, 5, 7, 8, ]");
}

TEST(TensorSlicing, RowSlice)
{
    std::vector<size_t> shape = { 3, 3 };
    size_t size = shape[0] * shape[1];
    int* data = new int[size];
    for (int i = 0; i < size; i++)
        data[i] = i;

    Tensor<int> tensor(data, shape);

    auto sub_tensor = tensor[{
        { 0, 1 },
        { 0, 3 }
    }];
    auto* sub_shape = sub_tensor.shape();

    EXPECT_EQ(sub_shape[0], 1);
    EXPECT_EQ(sub_shape[1], 3);
    EXPECT_STREQ(sub_tensor.flat_string().c_str(), "[0, 1, 2, ]");

    auto sub_tensor2 = tensor[{
        { 1, 2 },
        { 0, 3 }
    }];
    auto* sub_shape2 = sub_tensor2.shape();

    EXPECT_EQ(sub_shape2[0], 1);
    EXPECT_EQ(sub_shape2[1], 3);
    EXPECT_STREQ(sub_tensor2.flat_string().c_str(), "[3, 4, 5, ]");

    auto sub_tensor3 = tensor[{
        { 2, 3 },
        { 0, 3 }
    }];
    auto* sub_shape3 = sub_tensor3.shape();

    EXPECT_EQ(sub_shape3[0], 1);
    EXPECT_EQ(sub_shape3[1], 3);
    EXPECT_STREQ(sub_tensor3.flat_string().c_str(), "[6, 7, 8, ]");
}

TEST(TensorSlicing, ColumnSlice)
{
    std::vector<size_t> shape = { 3, 3 };
    size_t size = shape[0] * shape[1];
    int* data = new int[size];
    for (int i = 0; i < size; i++)
        data[i] = i;

    Tensor<int> tensor(data, shape);

    auto sub_tensor = tensor[{
        { 0, 3 },
        { 0, 1 }
    }];
    auto* sub_shape = sub_tensor.shape();

    EXPECT_EQ(sub_shape[0], 3);
    EXPECT_EQ(sub_shape[1], 1);
    EXPECT_STREQ(sub_tensor.flat_string().c_str(), "[0, 3, 6, ]");

    auto sub_tensor2 = tensor[{
        { 0, 3 },
        { 1, 2 }
    }];
    auto* sub_shape2 = sub_tensor2.shape();

    EXPECT_EQ(sub_shape2[0], 3);
    EXPECT_EQ(sub_shape2[1], 1);
    EXPECT_STREQ(sub_tensor2.flat_string().c_str(), "[1, 4, 7, ]");

    auto sub_tensor3 = tensor[{
        { 0, 3 },
        { 2, 3 }
    }];
    auto* sub_shape3 = sub_tensor3.shape();

    EXPECT_EQ(sub_shape3[0], 3);
    EXPECT_EQ(sub_shape3[1], 1);
    EXPECT_STREQ(sub_tensor3.flat_string().c_str(), "[2, 5, 8, ]");
}