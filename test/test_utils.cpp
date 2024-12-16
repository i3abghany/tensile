#include "test_utils.h"

#include <numeric>

size_t flat_size(const std::vector<size_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

Tensile::Tensor<int> create_tensor(const std::vector<size_t>& shape)
{
    size_t len = flat_size(shape);
    int* data = new int[len];

    for (size_t i = 0; i < len; i++)
        data[i] = i;

    return Tensile::Tensor<int>(data, shape);
}