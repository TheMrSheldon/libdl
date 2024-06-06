#include <iostream>

#include <dl/device.hpp>
#include <dl/tensor/math.hpp>
#include <dl/tensor/tensor.hpp>

int main(int argc, char* argv[]) {
	dl::Tensor tensora = dl::constant({1.0f, 2.0f, 3.0f, 4.0f});
	tensora->setRequiresGrad(true);

	dl::Tensor tensorb = dl::constant({2.0f, 2.0f, 3.0f, 3.0f});
	tensorb->setRequiresGrad(true);
	dl::Tensor tensord = nullptr;
	{
		auto tensorc = tensora * tensorb;
		// Important: move tensorc's ownership into dl::mean, otherwise it will be deleted before backward() can be
		// called.
		tensord = std::move(dl::mean(std::move(tensorc)));
	}
	std::cout << tensord << std::endl; // Expected: 6.75
	tensord->backward();
	std::cout << tensora->gradient() << std::endl; // Expected: (0.50, 0.50, 0.75, 0.75)
	return 0;
}