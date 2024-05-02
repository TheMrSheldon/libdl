#include <catch2/catch_test_macros.hpp>

#include <dl/tensor/math.hpp>
#include <dl/tensor/tensor.hpp>
#include <dl/tensor/tensorptr.hpp>

TEST_CASE("Basic Differentiation", "[Autodiff]") {
	{
		/*dl::TensorPtr tensora = {1.0f, 2.0f, 3.0f, 4.0f};
		tensora->setRequiresGrad(true);

		dl::TensorPtr tensorb = {2.0f, 2.0f, 3.0f, 3.0f};
		tensorb->setRequiresGrad(true);
		auto tensorc = tensora * tensorb;
		auto tensord = dl::mean(tensorc);

		REQUIRE(tensora == dl::TensorPtr{1.0f, 2.0f, 3.0f, 4.0f});
		REQUIRE(tensora->requiresGrad());
		REQUIRE(tensorb == dl::TensorPtr{2.0f, 2.0f, 3.0f, 3.0f});
		REQUIRE(tensorb->requiresGrad());*/
	}
}