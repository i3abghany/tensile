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
    auto slice = tensor["0:2, 0:2"];
    auto logger = Tensile::Log::get_default_logger();
    logger->log(slice.flat_string());
}