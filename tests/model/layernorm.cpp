#include <dl/model/layernorm.hpp>

#include <dl/tensor/math.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("LayerNorm", "[Model]") {
	{
		dl::LayerNorm norm({3});
		{
			auto input = dl::constant({{1.0f, 2.0f, 3.0f}});
			auto output = norm.forward(input);
			CHECK(dl::allclose(output, dl::constant({{-1.225f, 0, 1.225f}}), 1e-5, 1e-3));
		}
		{
			auto input = dl::constant({{4.0f, 4.0f, 6.0f}});
			auto output = norm.forward(input);
			CHECK(dl::allclose(output, dl::constant({{-0.707f, -0.707f, 1.414f}}), 1e-5, 1e-3));
		}
		{
			auto input = dl::constant({{1.0f, 2.0f, 3.0f}, {4.0f, 4.0f, 6.0f}});
			auto output = norm.forward(input);
			CHECK(dl::allclose(output, dl::constant({{-1.225f, 0, 1.225f}, {-0.707f, -0.707f, 1.414f}}), 1e-5, 1e-3));
		}
	}
}