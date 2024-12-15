#include "test_utils.h"

#include <numeric>

size_t flat_size(const std::vector<size_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}
