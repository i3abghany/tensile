#include <cassert>
#include <memory>

#include "logger.h"
#include "tensor.h"

int main()
{
    int* data = new int[9];
    std::vector<size_t> shape = { 3, 3 };
    for (int i = 0; i < 9; i++) {
        data[i] = i;
    }

    Tensile::Tensor<int> tensor(data, shape);

    shape = { 3, 1 };
    data = new int[3];
    for (int i = 0; i < 3; i++) {
        data[i] = i;
    }

    Tensile::Tensor<int> other(data, shape);

    auto result = tensor + other;
    Tensile::Log::get_ostream_logger()->log("Result: " + result.flat_string());

}