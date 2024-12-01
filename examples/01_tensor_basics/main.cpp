#include <dl/device.hpp>
#include <dl/tensor/math.hpp>
#include <dl/tensor/tensorptr.hpp>

#include <iostream>
#include <ranges>

int main(int argc, char* argv[]) {
	dl::TensorPtr tensora = dl::constant({1.0f, 2.0f, 3.0f, 4.0f});
	tensora->setRequiresGrad(true);

	dl::TensorPtr tensorb = dl::constant({2.0f, 2.0f, 3.0f, 3.0f});
	tensorb->setRequiresGrad(true);
	dl::TensorPtr tensord = nullptr;
	{
		auto tensorc = tensora * tensorb;
		tensord = dl::mean(tensorc);
	}
	std::cout << tensord << std::endl; // Expected: 6.75
	tensord->backward();
	std::cout << tensora->gradient() << std::endl; // Expected: (0.50, 0.50, 0.75, 0.75)

	// Iterating a tensor
	for (auto entry : tensora)
		std::cout << entry << std::endl;

	for (const auto& [left, right] : std::views::zip(tensora, tensorb))
		std::cout << left << " : " << right << std::endl;
	return 0;
}