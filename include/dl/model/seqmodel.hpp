#ifndef DL_MODEL_SEQMODEL_HPP
#define DL_MODEL_SEQMODEL_HPP

#ifndef DL_MODEL_MODEL_HPP
#error Include model.hpp instead of seqmodel.hpp
#endif

#include <functional>

namespace dl {

	template <typename M1, typename M2>
	class SeqModel final : public Model {
	private:
		M1 model1;
		M2 model2;

	public:
		SeqModel(M1&& m1, M2&& m2) noexcept : model1(std::move(m1)), model2(std::move(m2)) {}

		template <typename... Args>
		auto operator()(Args&&... args) {
			return model2(model1(std::forward<Args>(args)...));
		}
	};
} // namespace dl

namespace dl {
	auto Model::operator|(this auto self, auto right) -> SeqModel<decltype(self), decltype(right)> {
		return {std::move(self), std::move(right)};
	}
} // namespace dl

#endif