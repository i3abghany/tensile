#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace Tensile {

std::vector<std::pair<size_t, size_t>> parse_indices(const std::string& input);

};