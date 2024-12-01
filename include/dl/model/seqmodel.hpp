#ifndef DL_MODEL_SEQMODEL_HPP
#define DL_MODEL_SEQMODEL_HPP

#ifndef DL_MODEL_MODEL_HPP
#error Include model.hpp instead of seqmodel.hpp
#endif
#include <functional>

namespace dl {

	template <typename M1, typename M2>
	class SeqModel final : public Model<R(Args...)> {
	private:
		M1 model1;
		M2 model2;
		std::function<R(Args...)> fun;

	public:
		SeqModel(M1&& m1, M2&& m2)
				: fun([m1 = std::move(m1), m2 = std::move(m2)](Args&&... args) -> R {
					  return m2(m1(std::forward<Args>(args)...));
				  }) {}

		R forward(Args... args) override { return fun(std::forward<Args>(args)...); }
	};
} // namespace dl

namespace dl {
	template <typename R, typename... Args>
	template <typename R2>
	SeqModel<R2(Args...)> Model<R(Args...)>::operator|(Model<R2(R)>&& right) {
		return SeqModel<R2(Args...)>(std::move(*this), std::move(right));
	}
} // namespace dl

#endif