#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dl/device.hpp>
#include <dl/tensor/rearrange.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("Rearrange", "[Rearrange]") {
	{
		/** \todo implement rearrange **/
		// REQUIRE_THAT(dl::rearrange("a b c -> a 10 b c", dl::zeros({1, 2, 3}))->shape(), ReangeEquals());
		// a b c -> a 10 b c          <= unsqueeze(1).expand(1, 10)
		// a ... b c -> a ... 10 b c  <= unsqueeze(-2).expand(-2, 10)
		// a b c -> a (10 b) c        <= expand(1, 10)
		// a 1 b c -> a b c           <= unsqueeze(1)
		// (a b) c -> b a c           <= reshape(a, b, c).perm({b, a, c}})
		// a -> a/3                   <= chunk(3, 0)
		// a ... b -> a ... b/3       <= chunk(3, -1)
		// a b -> (a b)/3             <= reshape(a*b).chunk(3, 0)
	}
}