#include <cassert>
#include <memory>

#include "logger.h"
#include "tensor.h"

int main()
{
    int* data = new int[36];
    std::vector<size_t> shape = { 4, 3, 3 };
    for (int i = 0; i < 36; i++) {
        data[i] = i;
    }

    Tensor<int> tensor(data, shape);

    auto sub_tensor = tensor[{
        { 1, 2 },
        { 0, 3 },
        { 0, 3 }
    }];
    Logger<std::ostream>::get_ostream_logger()->log(sub_tensor.flat_string());
}