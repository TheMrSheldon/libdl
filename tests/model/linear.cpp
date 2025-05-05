#include <dl/model/linear.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("Linear", "[Model]") {
	{
		dl::Linear linear(4, 5, true);
		linear.weights() = dl::InitializerTensor<float>{
				{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}, {16, 17, 18, 19}
		};
		linear.bias() = dl::InitializerTensor<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
		{
			dl::TensorPtr input = dl::InitializerTensor<float>{1, 2, 3, 4};
			auto output = linear(input);
			//CHECK(dl::allclose(output, dl::constant({{-1.225f, 0, 1.225f}}), 1e-5, 1e-3));
		}
	}
}