#include <catch2/catch_test_macros.hpp>

#include <dl/device.hpp>
#include <dl/tensor/math.hpp>
#include <dl/tensor/tensorimpl.hpp>
#include <dl/tensor/tensorptr.hpp>

TEST_CASE("Basic Differentiation", "[Autodiff]") {
	{
		dl::TensorPtr tensora = dl::constant({1.0f, 2.0f, 3.0f, 4.0f});
		tensora->setRequiresGrad(true);

		dl::TensorPtr tensorb = dl::constant({2.0f, 2.0f, 3.0f, 3.0f});
		tensorb->setRequiresGrad(true);
		auto tensorc = tensora * tensorb;
		auto tensord = dl::mean(tensorc);

		REQUIRE(tensorc->requiresGrad());
		REQUIRE(tensord->requiresGrad());

		tensord->backward();
		REQUIRE(dl::allclose(tensora->grad, dl::constant({0.50f, 0.50f, 0.75f, 0.75f})));
		REQUIRE(dl::allclose(tensorb->grad, dl::constant({0.25f, 0.50f, 0.75f, 1.00f})));
	}
	{
		dl::TensorPtr tensora =
				dl::constant({{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}});
		tensora->setRequiresGrad(true);

		dl::TensorPtr tensorb = dl::constant({2.0f, 3.0f, 4.0f});
		tensorb->setRequiresGrad(true);

		auto tensorc = dl::mean(dl::matmul(tensorb, tensora));
		tensorc->backward();
		REQUIRE(dl::allclose(tensorc, dl::constant(66.5f)));
		REQUIRE(dl::allclose(
				tensora->grad,
				dl::constant({{0.50f, 0.50f, 0.50f, 0.50f}, {0.75f, 0.75f, 0.75f, 0.75f}, {1.0f, 1.0f, 1.0f, 1.0f}})
		));
		REQUIRE(dl::allclose(tensorb->grad, dl::constant({2.5f, 6.5f, 10.5f})));
	}
}