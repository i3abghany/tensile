#include "index_parser.h"

#include <sstream>
#include <string>
#include <vector>

namespace Tensile {

static std::pair<size_t, size_t> parse_range(const std::string& input)
{
    size_t colon_pos = input.find(':');
    if (colon_pos == std::string::npos)
        throw std::invalid_argument("Invalid range format");

    try {
        size_t start = std::stoul(input.substr(0, colon_pos));
        size_t end = std::stoul(input.substr(colon_pos + 1));

        return { start, end };
    } catch (const std::exception& e) {
        throw std::invalid_argument("Invalid range format");
    }
}

std::vector<std::pair<size_t, size_t>> parse_indices(const std::string& input)
{
    std::vector<std::pair<size_t, size_t>> indices;
    std::istringstream iss(input);
    std::string token;

    while (std::getline(iss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);

        indices.push_back(parse_range(token));
    }

    return indices;
}

}