#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dl/device.hpp>
#include <dl/tensor/math.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("Math", "[TensorPtr]") {
	SECTION("multiplication") {
		{
			auto tensorA = dl::constant({{1, 2, 3}, {4, 5, 6}});
			auto tensorB = dl::transpose(dl::TensorPtr(tensorA), {-1, -2});
			auto result = dl::matmul(tensorA, tensorB);
			REQUIRE_THAT(result->shape(), RangeEquals(std::vector{2, 2}));
			CHECK(dl::allclose(result, dl::constant({{14, 32}, {32, 77}})));
		}
		{
			// Batched multiplication:
			// Multiplying a 2x2x3 tensor with a 2x3x2 tensor should result in a 2x2x2 tensor
			auto tensorA = dl::constant({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {0, 1, 2}}});
			auto tensorB = dl::transpose(dl::TensorPtr(tensorA), {-1, -2});
			auto result = dl::matmul(tensorA, tensorB);
			CHECK_THAT(tensorA->shape(), RangeEquals(std::vector{2, 2, 3}));
			CHECK_THAT(tensorB->shape(), RangeEquals(std::vector{2, 3, 2}));
			REQUIRE_THAT(result->shape(), RangeEquals(std::vector{2, 2, 2}));
			CHECK(dl::allclose(result, dl::constant({{{14, 32}, {32, 77}}, {{194, 26}, {26, 5}}})));
		}
	}
	SECTION("arithmetics") {
		const auto tensor = dl::constant({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
		{
			auto powResult = dl::pow(tensor, 2.0f);
			CHECK(dl::allclose(
					powResult, dl::constant({{1.0f, 4.0f, 9.0f}, {16.0f, 25.0f, 36.0f}, {49.0f, 64.0f, 81.0f}})
			));
		}
		{
			auto expResult = dl::exp(tensor);
			CHECK(dl::allclose(
					expResult,
					dl::constant(
							{{2.718f, 7.389f, 20.086f}, {54.598f, 148.413f, 403.429f}, {1096.63f, 2980.96f, 8103.08f}}
					),
					1e-5, 1e-2
			));
		}
		{
			auto sqrtResult = dl::sqrt(tensor);
			CHECK(dl::allclose(
					sqrtResult, dl::constant({{1.0f, 1.414f, 1.732f}, {2.0f, 2.236f, 2.449f}, {2.646f, 2.828f, 3.0f}}),
					1e-5, 1e-2
			));
		}
		{
			auto rsqrtResult = dl::rsqrt(tensor);
			CHECK(dl::allclose(
					rsqrtResult,
					dl::constant({{1.0f, 0.707f, 0.577f}, {0.5f, 0.447f, 0.408f}, {0.378f, 0.354f, 0.333f}}), 1e-5, 1e-2
			));
		}
	}

	SECTION("statistics") {
		const auto tensor = dl::constant({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
		{
			auto meanResult1 = dl::mean(tensor);
			CHECK(dl::allclose(meanResult1, dl::constant(5)));

			auto meanResult2 = dl::mean(tensor, 0);
			CHECK(dl::allclose(meanResult2, dl::constant({4.0f, 5.0f, 6.0f})));

			auto meanResult3 = dl::mean(tensor, 1);
			CHECK(dl::allclose(meanResult3, dl::constant({2.0f, 5.0f, 8.0f})));
		}
		{
			auto sumResult1 = dl::sum(tensor);
			CHECK(dl::allclose(sumResult1, dl::constant(45.0f)));

			auto sumResult2 = dl::sum(tensor, 0);
			CHECK(dl::allclose(sumResult2, dl::constant({12.0f, 15.0f, 18.0f})));

			auto sumResult3 = dl::sum(tensor, 1);
			CHECK(dl::allclose(sumResult3, dl::constant({6.0f, 15.0f, 24.0f})));
		}
		{
			auto minResult1 = dl::min(tensor);
			CHECK(dl::allclose(minResult1, dl::constant(1)));

			auto minResult2 = dl::min(tensor, 0);
			CHECK(dl::allclose(minResult2, dl::constant({1.0f, 2.0f, 3.0f})));

			auto minResult3 = dl::min(tensor, 1);
			CHECK(dl::allclose(minResult3, dl::constant({1.0f, 4.0f, 7.0f})));
		}
		{
			auto maxResult1 = dl::max(tensor);
			CHECK(dl::allclose(maxResult1, dl::constant(9.0f)));

			auto maxResult2 = dl::max(tensor, 0);
			CHECK(dl::allclose(maxResult2, dl::constant({7.0f, 8.0f, 9.0f})));

			auto maxResult3 = dl::max(tensor, 1);
			CHECK(dl::allclose(maxResult3, dl::constant({3.0f, 6.0f, 9.0f})));

			// keepdim
			auto maxResult4 = dl::max(tensor, 0, true);
			CHECK(dl::allclose(maxResult2, dl::constant({7.0f, 8.0f, 9.0f})));

			auto maxResult5 = dl::max(tensor, 1, true);
			CHECK(dl::allclose(maxResult3, dl::constant({3.0f, 6.0f, 9.0f})));

			// keepdim and 3D
			auto tensor3D = dl::constant({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
			CHECK_THAT(tensor3D->shape(), RangeEquals(std::vector{2, 2, 3}));
			auto maxResult6 = dl::max(tensor3D, 0, true);
			CHECK_THAT(maxResult6->shape(), RangeEquals(std::vector{1, 2, 3}));
			CHECK(dl::allclose(maxResult6, dl::constant({{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}})));

			auto maxResult7 = dl::max(tensor3D, 1, true);
			CHECK_THAT(maxResult7->shape(), RangeEquals(std::vector{2, 1, 3}));
			CHECK(dl::allclose(maxResult7, dl::constant({{{4.0f, 5.0f, 6.0f}}, {{10.0f, 11.0f, 12.0f}}})));

			auto maxResult8 = dl::max(tensor3D, 2, true);
			CHECK_THAT(maxResult8->shape(), RangeEquals(std::vector{2, 2, 1}));
			CHECK(dl::allclose(maxResult8, dl::reshape(dl::constant({{3.0f, 6.0f}, {9.0f, 12.0f}}), {2, 2, 1})));
		}
		{
			/** \todo these currently fail and crash and I don't know why **/
			auto varResult1 = dl::var(tensor);
			CHECK(dl::allclose(varResult1, dl::constant(7.5f)));

			/*auto varResult2 = dl::var(tensor, 0);
			CHECK(dl::allclose(varResult2, dl::constant({9.0f, 9.0f, 9.0f})));

			auto varResult3 = dl::var(tensor, 1);
			CHECK(dl::allclose(varResult3, dl::constant({1.0f, 1.0f, 1.0f})));*/
		}
		{
			const auto input = dl::constant({{-1, 2}, {-.5, 0}, {7, -3}, {4, -4}});
			auto reluResult = dl::relu(input);
			CHECK(dl::allclose(reluResult, dl::constant({{0, 2}, {0, 0}, {7, 0}, {4, 0}})));
		}
		{
			// No dim
			const auto input1 = dl::constant({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
			auto softmaxResult1 = dl::softmax(input1);
			CHECK(dl::allclose(softmaxResult1, dl::constant({0.0117f, 0.0317f, 0.086f, 0.234f, 0.636f}), 1e-5, 1e-2));

			const auto input2 = dl::constant({12345.0f, 67890.0f, 99999999.0f});
			auto softmaxResult2 = dl::softmax(input2);
			CHECK(dl::allclose(softmaxResult2, dl::constant({0, 0, 1}), 1e-5, 1e-2));

			// dim and 2D
			const auto input3 = dl::constant({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
			auto softmaxResult3 = dl::softmax(input3, 0);
			CHECK(dl::allclose(
					softmaxResult3, dl::constant({{0.047f, 0.047f, 0.047f}, {0.953f, 0.953f, 0.953f}}), 1e-5, 1e-2
			));
			auto softmaxResult4 = dl::softmax(input3, 1);
			CHECK(dl::allclose(
					softmaxResult4, dl::constant({{0.090f, 0.245f, 0.665f}, {0.090f, 0.245f, 0.665f}}), 1e-5, 1e-2
			));

			// dim and 3D
			const auto input4 =
					dl::constant({{{0.f, 1.f, 2.f}, {3.f, 4.f, 5.f}}, {{6.f, 7.f, 8.f}, {9.f, 10.f, 11.f}}});
			auto softmaxResult5 = dl::softmax(input4, 0);
			auto softmaxResult6 = dl::softmax(input4, 1);
			auto softmaxResult7 = dl::softmax(input4, 2);
			CHECK(dl::allclose(
					softmaxResult5,
					dl::constant(
							{{{0.0025, 0.0025, 0.0025}, {0.0025, 0.0025, 0.0025}},
							 {{0.9975, 0.9975, 0.9975}, {0.9975, 0.9975, 0.9975}}}
					),
					1e-5, 1e-2
			));
			CHECK(dl::allclose(
					softmaxResult6,
					dl::constant(
							{{{0.0474, 0.0474, 0.0474}, {0.9526, 0.9526, 0.9526}},
							 {{0.0474, 0.0474, 0.0474}, {0.9526, 0.9526, 0.9526}}}
					),
					1e-5, 1e-2
			));
			CHECK(dl::allclose(
					softmaxResult7,
					dl::constant(
							{{{0.0900, 0.2447, 0.6652}, {0.0900, 0.2447, 0.6652}},
							 {{0.0900, 0.2447, 0.6652}, {0.0900, 0.2447, 0.6652}}}
					),
					1e-5, 1e-2
			));
		}
	}
}