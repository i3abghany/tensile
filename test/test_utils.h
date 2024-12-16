#pragma once

#include <cstdint>
#include <vector>

#include "tensor.h"

size_t flat_size(const std::vector<size_t>& shape);
Tensile::Tensor<int> create_tensor(const std::vector<size_t>& shape);