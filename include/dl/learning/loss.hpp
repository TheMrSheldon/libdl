#pragma once

#include "../tensor/math.hpp"

namespace dl::loss {
    TensorPtr mse(TensorPtr x, TensorPtr y) noexcept {
        return mean(pow(x-y, 2));
    }
}