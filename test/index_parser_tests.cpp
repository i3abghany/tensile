#include <gtest/gtest.h>

#include "index_parser.h"

TEST(ParseIndices, EmptyString)
{
    std::string input = "";
    auto res = parse_indices(input);
    EXPECT_EQ(res.size(), 0);
}

class ParseIndicesParametrized
    : public ::testing::TestWithParam<std::pair<std::string, std::vector<std::pair<size_t, size_t>>>> { };

TEST_P(ParseIndicesParametrized, ParseIndices)
{
    auto& [input, expected] = GetParam();
    auto res = parse_indices(input);
    EXPECT_EQ(res, expected);
}

INSTANTIATE_TEST_SUITE_P(
    ParseIndices, ParseIndicesParametrized,
    ::testing::Values(
        std::make_pair("0:1",
                       std::vector<std::pair<size_t, size_t>> {
                           { 0, 1 }
}),
        std::make_pair("0:1, 2:3", std::vector<std::pair<size_t, size_t>> { { 0, 1 }, { 2, 3 } }),
        std::make_pair("0:1, 2:3", std::vector<std::pair<size_t, size_t>> { { 0, 1 }, { 2, 3 } }),
        std::make_pair("0:1, 2:3, 4:5", std::vector<std::pair<size_t, size_t>> { { 0, 1 }, { 2, 3 }, { 4, 5 } }),
        std::make_pair("0:1, 2:3, 4:5, 6:7",
                       std::vector<std::pair<size_t, size_t>> { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 } }),
        std::make_pair("0:1, 2:3, 4:5, 6:7, 8:9",
                       std::vector<std::pair<size_t, size_t>> { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 8, 9 } }),
        std::make_pair("0:1   ,2:3,  4:5,6:7,   8:9",
                       std::vector<std::pair<size_t, size_t>> { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 8, 9 } })));

class ParseIndicesInvalid : public ::testing::TestWithParam<std::string> { };

TEST_P(ParseIndicesInvalid, ParseIndices)
{
    auto input = GetParam();
    EXPECT_THROW(parse_indices(input), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(ParseIndices, ParseIndicesInvalid,
                         ::testing::Values("0", "0:", ":1",
                                           "0:1,2:", "0:1,2:3,4:", "0:1,2:3,4:5,6:", "0:1,2:3,4:5,6:7,8:"));