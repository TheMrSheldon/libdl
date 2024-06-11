/**
 * @file adapters.hpp
 * @brief Contains some general purpose loss adapters.
 */

#ifndef DL_LEARNING_ADAPTERS_HPP
#define DL_LEARNING_ADAPTERS_HPP

#include <concepts>

namespace dl {

	/**
     * @brief A training adapter for applying a single loss objective.
     * @details Let \f(\mathcal M\colon X\to Y\f) denote the model to be trained or evaluated, where \f(X\f) is the
     * space of all inputs and \f(Y\f) is the space of all outputs. Further, let
     * \f(\mathcal L\colon Y\times Y \to \mathbb R\f) denote a loss objective. Given the input \f(x\in X\f) and desired
     * output \f(y\in Y\f), the loss adapter returns
     * \f[\mathcal{L}(\mathcal{M}(x), y).\f]
     * Or, in short: The loss adapter tells the trainer to optimize the given loss objective.
     * 
     * @param lossObjective the loss objective to be optimized for.
     * @return A loss adapter for optimizing the specified loss objective.
     */
	auto lossAdapter(auto lossObjective) {
		constexpr auto func = [](auto loss, auto& model, auto& x, auto& y) { return loss(model(x), y); };
		return std::bind_front(func, lossObjective);
	}

} // namespace dl

#endif
