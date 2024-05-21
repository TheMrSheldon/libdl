#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dl/device.hpp>
#include <dl/tensor/tensorimpl.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("Basics", "[Tensor]") {
	{
		dl::InitializerTensor<float> init1D = {1, 2, 3};
		CHECK_THAT(init1D.shape, RangeEquals(std::vector{3}));

		dl::InitializerTensor<float> init2D = {{1, 2, 3}, {4, 5, 6}};
		CHECK_THAT(init2D.shape, RangeEquals(std::vector{2, 3}));
	}
	{
		auto tensor = dl::empty({1, 2, 3});
		CHECK(tensor->shape(0) == 1);
		CHECK(tensor->shape(1) == 2);
		CHECK(tensor->shape(2) == 3);
		CHECK_THAT(tensor->shape(), RangeEquals(std::vector{1, 2, 3}));
	}
	{
		auto tensor = dl::ones({4, 3});
		CHECK(tensor->shape(0) == 4);
		CHECK(tensor->shape(1) == 3);
		CHECK_THAT(tensor->shape(), RangeEquals(std::vector{4, 3}));
		auto flat = tensor->flatten();
		CHECK(flat->shape(0) == 12);
		CHECK_THAT(flat->shape(), RangeEquals(std::vector{12}));
		CHECK(flat == dl::constant({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
	}
	{
		auto tensorA = dl::constant({{1, 2, 3}, {4, 5, 6}});
		auto tensorB = dl::constant({{7, 8, 9}, {0, 1, 2}});
		auto tensorC = dl::constant({{7, 8, 9}});
		auto tensorD = dl::constant({7, 8});
		tensorD->reshape({2, 1});
		CHECK(dl::allclose(tensorA * tensorB, dl::constant({{7, 16, 27}, {0, 5, 12}})));
		CHECK(dl::allclose(tensorA * tensorC, dl::constant({{7, 16, 27}, {28, 40, 54}})));
		CHECK(dl::allclose(tensorA * tensorD, dl::constant({{7, 14, 21}, {32, 40, 48}})));
	}
}