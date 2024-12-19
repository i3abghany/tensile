#include <gtest/gtest.h>

#include "tensile/tensor.h"
#include "test_utils.h"

using std::array;
using std::make_tuple;
using std::pair;
using std::string;
using std::tuple;
using std::vector;
using Tensile::Tensor;

class TensorElementwiseShapeCompatibilityTest
    : public ::testing::TestWithParam<tuple<vector<size_t>, vector<size_t>, bool>> {
protected:
    Tensor<int>* tensor_a;
    Tensor<int>* tensor_b;
    vector<size_t> shape_a;
    vector<size_t> shape_b;
    size_t len_a;
    size_t len_b;

    bool legal;

    void SetUp() override
    {
        shape_a = std::get<0>(GetParam());
        shape_b = std::get<1>(GetParam());
        legal = std::get<2>(GetParam());
        len_a = flat_size(shape_a);
        len_b = flat_size(shape_b);

        tensor_a = new Tensor<int>(nullptr, shape_a);
        tensor_b = new Tensor<int>(nullptr, shape_b);
    }

    void TearDown() override
    {
        delete tensor_a;
        delete tensor_b;
    }
};

TEST_P(TensorElementwiseShapeCompatibilityTest, ShapeCompatibility)
{
    if (legal)
        ASSERT_TRUE(Tensor<int>::shape_compat(*tensor_a, *tensor_b));
    else
        ASSERT_FALSE(Tensor<int>::shape_compat(*tensor_a, *tensor_b));
}

INSTANTIATE_TEST_SUITE_P(
    TensorElementwiseShapeCompatibilityTest, TensorElementwiseShapeCompatibilityTest,
    ::testing::Values(make_tuple(vector<size_t> { 3 }, vector<size_t> { 3 }, true),
                      make_tuple(vector<size_t> { 3 }, vector<size_t> { 1 }, true),
                      make_tuple(vector<size_t> { 1 }, vector<size_t> { 3 }, true),
                      make_tuple(vector<size_t> { 1 }, vector<size_t> { 1 }, true),
                      make_tuple(vector<size_t> { 1, 3 }, vector<size_t> { 3, 1 }, true),
                      make_tuple(vector<size_t> { 1, 3 }, vector<size_t> { 1, 3 }, true),
                      make_tuple(vector<size_t> { 3, 1 }, vector<size_t> { 3, 1 }, true),
                      make_tuple(vector<size_t> { 3, 1 }, vector<size_t> { 1, 3 }, true),
                      make_tuple(vector<size_t> { 3, 1, 1 }, vector<size_t> { 3, 5, 1 }, true),
                      make_tuple(vector<size_t> { 3, 1, 1 }, vector<size_t> { 3, 1, 1 }, true),
                      make_tuple(vector<size_t> { 3, 1, 1 }, vector<size_t> { 3, 1, 5 }, true),
                      make_tuple(vector<size_t> { 3, 1, 1 }, vector<size_t> { 3, 5, 5 }, true),
                      make_tuple(vector<size_t> { 3, 1, 1 }, vector<size_t> { 1, 5, 5 }, true),
                      make_tuple(vector<size_t> { 3, 1, 1 }, vector<size_t> { 1, 1, 5 }, true),
                      make_tuple(vector<size_t> { 3, 1, 1 }, vector<size_t> { 1, 5, 1 }, true),
                      make_tuple(vector<size_t> { 1, 1, 1, 3 }, vector<size_t> { 3, 3, 3, 3 }, true),
                      make_tuple(vector<size_t> { 3, 1, 1 }, vector<size_t> { 1, 1, 2 }, true),
                      make_tuple(vector<size_t> { 4 }, vector<size_t> { 1 }, true),
                      make_tuple(vector<size_t> { 1, 4, 2, 2 }, vector<size_t> { 4, 1, 2, 1 }, true),
                      //   make_tuple(vector<size_t> { 2, 3, 3 }, vector<size_t> { 3, 1 }, true), // FIXME
                      //   make_tuple(vector<size_t> { 4, 2, 2, 3 }, vector<size_t> { 1, 1 }, true), // FIXME

                      make_tuple(vector<size_t> { 3 }, vector<size_t> { 5 }, false),
                      make_tuple(vector<size_t> { 1, 2 }, vector<size_t> { 1, 1, 3 }, false),
                      make_tuple(vector<size_t> { 3 }, vector<size_t> { 2, 2 }, false),
                      make_tuple(vector<size_t> { 3, 2, 3 }, vector<size_t> { 3, 4, 2 }, false),
                      make_tuple(vector<size_t> { 3, 2, 2 }, vector<size_t> { 2, 3, 2, 4 }, false),
                      make_tuple(vector<size_t> { 4, 2, 1, 2 }, vector<size_t> { 4, 3 }, false),
                      make_tuple(vector<size_t> { 2, 4, 2, 4 }, vector<size_t> { 2, 4, 4 }, false),
                      make_tuple(vector<size_t> { 4, 1, 1, 3 }, vector<size_t> { 4, 2 }, false),
                      make_tuple(vector<size_t> { 4, 1, 2 }, vector<size_t> { 2, 2, 3 }, false),
                      make_tuple(vector<size_t> { 4, 3, 1, 2 }, vector<size_t> { 2, 4, 2, 2 }, false),
                      make_tuple(vector<size_t> { 2, 3 }, vector<size_t> { 1, 1, 2 }, false),
                      make_tuple(vector<size_t> { 3, 3, 3, 4 }, vector<size_t> { 2, 1, 1, 4 }, false),
                      make_tuple(vector<size_t> { 4, 3, 4, 4 }, vector<size_t> { 2 }, false),
                      make_tuple(vector<size_t> { 3, 1 }, vector<size_t> { 2, 1 }, false),
                      make_tuple(vector<size_t> { 4, 4, 1 }, vector<size_t> { 2, 3, 4, 2 }, false),
                      make_tuple(vector<size_t> { 2, 3, 2 }, vector<size_t> { 2, 3, 4 }, false),
                      make_tuple(vector<size_t> { 4, 1, 2 }, vector<size_t> { 1, 1, 1, 3 }, false),
                      make_tuple(vector<size_t> { 4, 4, 1 }, vector<size_t> { 3, 1 }, false),
                      make_tuple(vector<size_t> { 1, 3 }, vector<size_t> { 3, 4 }, false),
                      make_tuple(vector<size_t> { 4, 1, 3, 1 }, vector<size_t> { 2, 4, 2 }, false),
                      make_tuple(vector<size_t> { 1, 2, 1 }, vector<size_t> { 2, 3, 3, 1 }, false),
                      make_tuple(vector<size_t> { 3, 1, 1, 3 }, vector<size_t> { 2 }, false),
                      make_tuple(vector<size_t> { 1, 4 }, vector<size_t> { 2 }, false),
                      make_tuple(vector<size_t> { 2, 4, 1 }, vector<size_t> { 3, 4, 1 }, false),
                      make_tuple(vector<size_t> { 4, 4, 3, 4 }, vector<size_t> { 4, 3 }, false),
                      make_tuple(vector<size_t> { 1, 2, 4 }, vector<size_t> { 4, 1, 2 }, false),
                      make_tuple(vector<size_t> { 1, 3 }, vector<size_t> { 2, 4 }, false),
                      make_tuple(vector<size_t> { 3, 3, 3 }, vector<size_t> { 2, 3, 2 }, false),
                      make_tuple(vector<size_t> { 3, 4 }, vector<size_t> { 4, 1 }, false),
                      make_tuple(vector<size_t> { 1, 1, 1, 3 }, vector<size_t> { 4, 2 }, false),
                      make_tuple(vector<size_t> { 1, 2, 1, 4 }, vector<size_t> { 2, 1, 2 }, false),
                      make_tuple(vector<size_t> { 2, 4, 4, 2 }, vector<size_t> { 4, 3, 3 }, false)));

class TensorMatmul2DShapeCompatibilityTest
    : public ::testing::TestWithParam<tuple<array<size_t, 2>, array<size_t, 2>, bool>> {
protected:
    Tensor<int>* tensor_a;
    Tensor<int>* tensor_b;
    array<size_t, 2> shape_a;
    array<size_t, 2> shape_b;
    size_t len_a;
    size_t len_b;

    bool legal;

    void SetUp() override
    {
        shape_a = std::get<0>(GetParam());
        shape_b = std::get<1>(GetParam());
        legal = std::get<2>(GetParam());
        len_a = flat_size(vector<size_t>(shape_a.begin(), shape_a.end()));
        len_a = flat_size(vector<size_t>(shape_a.begin(), shape_a.end()));

        tensor_a = new Tensor<int>(nullptr, shape_a);
        tensor_b = new Tensor<int>(nullptr, shape_b);
    }

    void TearDown() override
    {
        delete tensor_a;
        delete tensor_b;
    }
};

TEST_P(TensorMatmul2DShapeCompatibilityTest, ShapeCompatibility)
{
    if (legal)
        ASSERT_TRUE(Tensor<int>::matmul_compat(*tensor_a, *tensor_b));
    else
        ASSERT_FALSE(Tensor<int>::matmul_compat(*tensor_a, *tensor_b));
}

constexpr auto generate_test_cases()
{
    constexpr size_t M = 5;
    constexpr size_t num_cases = M * M * M * M;
    array<tuple<array<size_t, 2>, array<size_t, 2>, bool>, num_cases> test_cases;

    size_t index = 0;
    for (size_t i = 1; i <= M; i++) {
        for (size_t j = 1; j <= M; j++) {
            for (size_t k = 1; k <= M; k++) {
                for (size_t l = 1; l <= M; l++) {
                    array<size_t, 2> shape_a = { i, j };
                    array<size_t, 2> shape_b = { k, l };
                    bool legal = j == k;
                    test_cases[index++] = make_tuple(shape_a, shape_b, legal);
                }
            }
        }
    }
    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(TensorMatmul2DShapeCompatibilityTest, TensorMatmul2DShapeCompatibilityTest,
                         ::testing::ValuesIn(generate_test_cases()));
