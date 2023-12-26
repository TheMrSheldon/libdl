#pragma once

#include <concepts>
#include <memory>
#include <vector>

namespace dl {
    class Tensor;
	using Shape = std::vector<std::size_t>;
    class TensorPtr;

    class TensorPtr : std::shared_ptr<Tensor> {
    private:
        /** \todo add autograd graph **/
    public:
        TensorPtr(std::shared_ptr<Tensor>&& other) : std::shared_ptr<Tensor>(std::move(other)) {};
        TensorPtr(const std::shared_ptr<Tensor>& other) : std::shared_ptr<Tensor>(other) {};
        TensorPtr(TensorPtr&& other) : std::shared_ptr<Tensor>(std::move(other)) {};
        TensorPtr(const TensorPtr& other) : std::shared_ptr<Tensor>(other) {};

        TensorPtr(std::integral auto value);
        TensorPtr(std::floating_point auto value);

        using std::shared_ptr<Tensor>::operator bool;
        using std::shared_ptr<Tensor>::operator*;
        using std::shared_ptr<Tensor>::operator->;
        using std::shared_ptr<Tensor>::operator=;

        TensorPtr& operator=(const TensorPtr& other) {
            this->std::shared_ptr<Tensor>::operator=(other);
            return *this;
        }
        TensorPtr& operator=(TensorPtr&& other) {
            this->std::shared_ptr<Tensor>::operator=(std::move(other));
            return *this;
        }
    };
}

#include "../device.hpp"
dl::TensorPtr::TensorPtr(std::integral auto value) : dl::TensorPtr(std::move(constant(value))) {}
dl::TensorPtr::TensorPtr(std::floating_point auto value) : dl::TensorPtr(std::move(constant(value))) {}