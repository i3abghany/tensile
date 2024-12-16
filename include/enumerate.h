#include "tensor.h"

#define ENUMERATE(iter_shape, i, j, k, l)                                                                              \
    for (size_t i = 0; i < iter_shape[0]; i++)                                                                         \
        for (size_t j = 0; j < iter_shape[1]; j++)                                                                     \
            for (size_t k = 0; k < iter_shape[2]; k++)                                                                 \
                for (size_t l = 0; l < iter_shape[3]; l++)
