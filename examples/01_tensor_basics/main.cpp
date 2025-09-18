#include <dl/device.hpp>
#include <dl/tensor/math.hpp>
#include <dl/tensor/rearrange.hpp>
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

	// Rearrange
	// dl::rearrange(specstr, tensor)
	// a b c -> a 10 b c          <= unsqueeze(1).expand(1, 10)
	// a ... b c -> a ... 10 b c  <= unsqueeze(-2).expand(-2, 10)
	// a b c -> a (10 b) c        <= expand(1, 10)
	// a 1 b c -> a b c           <= unsqueeze(1)
	// (a b) c -> b a c           <= reshape(a, b, c).perm({b, a, c}})
	// a -> a/3                   <= chunk(3, 0)
	// a ... b -> a ... b/3       <= chunk(3, -1)
	// a b -> (a b)/3             <= reshape(a*b).chunk(3, 0)

	// std::cout << dl::rearrange({"(a b) c -> b a c", {{"a", 10}, {"b", 5}}}, tensora) << std::endl;

	return 0;
}