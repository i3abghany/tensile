#include <gtest/gtest.h>

#include "tensile/tensor.h"
#include "test_utils.h"

using Ind = std::vector<size_t>;

TEST(Tensor3dMatmulTest, Matmul)
{
    auto t1 = create_tensor({ 4, 2, 2 });
    auto t2 = create_tensor({ 4, 2, 2 });

    auto result = t1 * t2;

    const auto r0 = result["0:1, 0:2, 0:2"];
    ASSERT_EQ((r0[Ind { 0, 0, 0 }]), 2);
    ASSERT_EQ((r0[Ind { 0, 0, 1 }]), 3);
    ASSERT_EQ((r0[Ind { 0, 1, 0 }]), 6);
    ASSERT_EQ((r0[Ind { 0, 1, 1 }]), 11);

    const auto r1 = result["1:2, 0:2, 0:2"];
    ASSERT_EQ((r1[Ind { 0, 0, 0 }]), 46);
    ASSERT_EQ((r1[Ind { 0, 0, 1 }]), 55);
    ASSERT_EQ((r1[Ind { 0, 1, 0 }]), 66);
    ASSERT_EQ((r1[Ind { 0, 1, 1 }]), 79);

    const auto r2 = result["2:3, 0:2, 0:2"];
    ASSERT_EQ((r2[Ind { 0, 0, 0 }]), 154);
    ASSERT_EQ((r2[Ind { 0, 0, 1 }]), 171);
    ASSERT_EQ((r2[Ind { 0, 1, 0 }]), 190);
    ASSERT_EQ((r2[Ind { 0, 1, 1 }]), 211);

    const auto r3 = result["3:4, 0:2, 0:2"];
    ASSERT_EQ((r3[Ind { 0, 0, 0 }]), 326);
    ASSERT_EQ((r3[Ind { 0, 0, 1 }]), 351);
    ASSERT_EQ((r3[Ind { 0, 1, 0 }]), 378);
    ASSERT_EQ((r3[Ind { 0, 1, 1 }]), 407);
}
