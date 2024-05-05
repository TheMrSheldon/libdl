#include <catch2/catch_test_macros.hpp>

#include <dl/tensor/math.hpp>
#include <dl/tensor/tensor.hpp>
#include <dl/tensor/tensorimpl.hpp>

TEST_CASE("Basic Differentiation", "[Autodiff]") {
	{
		/*dl::Tensor tensora = {1.0f, 2.0f, 3.0f, 4.0f};
		tensora->setRequiresGrad(true);

		dl::Tensor tensorb = {2.0f, 2.0f, 3.0f, 3.0f};
		tensorb->setRequiresGrad(true);
		auto tensorc = tensora * tensorb;
		auto tensord = dl::mean(tensorc);

		REQUIRE(tensora == dl::Tensor{1.0f, 2.0f, 3.0f, 4.0f});
		REQUIRE(tensora->requiresGrad());
		REQUIRE(tensorb == dl::Tensor{2.0f, 2.0f, 3.0f, 3.0f});
		REQUIRE(tensorb->requiresGrad());*/
	}
}