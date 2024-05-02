#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dl/device.hpp>
#include <dl/tensor/tensor.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("Basics", "[Tensor]") {
	{
		auto tensor = dl::empty({1, 2, 3});
		REQUIRE(tensor->shape(0) == 1);
		REQUIRE(tensor->shape(1) == 2);
		REQUIRE(tensor->shape(2) == 3);
		REQUIRE_THAT(tensor->shape(), RangeEquals(std::vector{1, 2, 3}));
	}
	{
		auto tensor = dl::empty({4, 3});
		REQUIRE(tensor->shape(0) == 4);
		REQUIRE(tensor->shape(1) == 3);
		REQUIRE_THAT(tensor->shape(), RangeEquals(std::vector{4, 3}));
		auto flat = tensor->flatten();
		REQUIRE(flat->shape(0) == 12);
		REQUIRE_THAT(flat->shape(), RangeEquals(std::vector{12}));
		// REQUIRE(flat == {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
		REQUIRE(flat == dl::constant({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
		// REQUIRE_THAT(flat, RangeEquals(std::vector{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
	}
	{
		auto tensor = dl::constant({2, 4, 5, 1, 3});
		REQUIRE_THAT(tensor->shape(), RangeEquals(std::vector{5}));
		auto mean = tensor->mean();
		REQUIRE(mean->shape().empty());
		// REQUIRE(mean == {3});
		REQUIRE(mean == dl::constant(3));
		// REQUIRE_THAT(mean, RangeEquals(std::vector{3}));
	}
}