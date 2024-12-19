#include <cassert>
#include <memory>

#include "tensile/logger.h"
#include "tensile/tensor.h"

int main()
{
    std::vector<size_t> s1 = { 3, 4 }, s2 = { 4, 3 };
    auto get_data = [](size_t i, size_t j) {
        int* data = new int[i * j];
        std::iota(data, data + i * j, 1);
        return data;
    };
    auto d1 = get_data(s1[0], s1[1]), d2 = get_data(s2[0], s2[1]);
    Tensile::Tensor<int> t1(d1, s1), t2(d2, s2);

    auto result = t1 * t2; // matmul
    Tensile::Log::get_ostream_logger()->log("t1: " + t1.to_string() + "\n");
    Tensile::Log::get_ostream_logger()->log("t2: " + t2.to_string() + "\n");
    Tensile::Log::get_ostream_logger()->log("Result: " + result.to_string());
}