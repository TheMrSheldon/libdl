#ifndef DL_TENSOR_REARRANGE_HPP
#define DL_TENSOR_REARRANGE_HPP

#include "tensorptr.hpp"

namespace dl {

	namespace detail {
		// This is the runtime rearrange spec for now
		struct RearrangeSpec final {
		private:
		public:
			RearrangeSpec(std::string_view str, std::initializer_list<std::tuple<std::string_view, size_t>> dims) {}
			constexpr bool needsSqueeze() const;

			constexpr bool needsUnsqueeze() const;
			constexpr bool needsRearrange() const;
			constexpr bool needsTranspose() const;
		};

		/* // This was the first draw towards compile-time evaluating the rearrange spec
		class RearrangeSpec {
		private:
			static constexpr ctll::fixed_string fullexpr = "^(?<left>.*?)\\s*->\\s*(?<right>.*?)$";
			// static constexpr ctll::fixed_string sideexpr = "^(?:(?:\\((?:\\s*\\w+\\s*?)+\\)|\\w+)(?:\\/\\d+)?\\s*)+?$";
			static constexpr ctll::fixed_string nameddimexpr =
					"[A-Za-z]+"; //"[^\\W\\d\\s]+"; //<-- this regex may work better with unicode but leads to compile errors for now
			static constexpr ctll::fixed_string leftstr = "left";
			static constexpr ctll::fixed_string rightstr = "right";

			template <typename Iterator, typename... Captures>
			static constexpr auto splitConf(const ctre::regex_results<Iterator, Captures...>& match) {
				//static_assert(match.get<0>(), "Invalid rearrange specification");
				//static_assert(ctre::match<sideexpr>(match.get<leftstr>()), "Invalid left rearrange specification");
				//static_assert(ctre::match<sideexpr>(match.get<rightstr>()), "Invalid right rearrange specification");
				return std::make_tuple(
						match.template get<leftstr>().to_view(), match.template get<rightstr>().to_view()
				);
			}
			static constexpr auto readConfiguration(const std::string_view str) {
				return splitConf(ctre::match<fullexpr>(str));
			}

			template <typename BeginIterator, typename EndIterator, typename RE, typename ResultIterator>
			static constexpr size_t
			numMatches(ctre::regex_iterator<BeginIterator, EndIterator, RE, ResultIterator> it) {
				return (it == ctre::regex_end_iterator{}) ? 0 : (1 + numMatches(++it));
			}
			static constexpr size_t numMatches(const auto& range) { return numMatches(range.begin()); }
			const std::tuple<std::string_view, std::string_view> spec;
			const size_t numNamedDims;

		public:
			constexpr RearrangeSpec(std::string_view spec)
					: spec(readConfiguration(spec)),
					  numNamedDims(numMatches(ctre::search_all<nameddimexpr>(std::get<0>(this->spec)))) {}

			constexpr size_t numNamedDimensions() const noexcept { return numNamedDims; }
		};

		constexpr auto permute(const auto tmp, dl::TensorPtr& tensor) {}

		template <typename Iterator, typename... Captures>
		constexpr auto splitConf(const ctre::regex_results<Iterator, Captures...>& match) {
			// constexpr ctll::fixed_string sideexpr = "^(?:(?:\\((?:\\s*\\w+\\s*?)+\\)|\\w+)(?:\\/\\d+)?\\s*)+?$";
			constexpr ctll::fixed_string leftstr = "left";
			constexpr ctll::fixed_string rightstr = "right";
			//static_assert(match.get<0>(), "Invalid rearrange specification");
			//static_assert(ctre::match<sideexpr>(match.get<leftstr>()), "Invalid left rearrange specification");
			//static_assert(ctre::match<sideexpr>(match.get<rightstr>()), "Invalid right rearrange specification");
			return std::make_tuple(match.template get<leftstr>().to_view(), match.template get<rightstr>().to_view());
		}

		constexpr auto readConfiguration(const std::string_view str) {
			constexpr ctll::fixed_string fullexpr = "^(?<left>.*?)\\s*->\\s*(?<right>.*?)$";
			return splitConf(ctre::match<fullexpr>(str));
		}

		template <typename BeginIterator, typename EndIterator, typename RE, typename ResultIterator>
		constexpr size_t numMatches(ctre::regex_iterator<BeginIterator, EndIterator, RE, ResultIterator> it) {
			return (it == ctre::regex_end_iterator{}) ? 0 : (1 + numMatches(++it));
		}

		constexpr size_t numMatches(const auto& range) { return numMatches(range.begin()); }*/
	} // namespace detail

	/**
	 * @brief 
	 * @details
	 * a b c -> a 10 b c  <= unsqueeze(1).expand(1, 10)
	 * a ... b c -> a ... 10 b c  <= unsqueeze(-2).expand(-2, 10)
	 * a b c -> a (10 b) c  <= expand(1, 10)
	 * a 1 b c -> a b c  <= unsqueeze(1)
	 * (a b) c -> b a c  <= reshape(a, b, c).transpose({{a, b}, {c}})
	 * 
	 * @param spec 
	 * @param tensor 
	 * @return TensorPtr 
	 */
	[[nodiscard]] TensorPtr rearrange(detail::RearrangeSpec spec, TensorPtr tensor) noexcept;

} // namespace dl

#endif