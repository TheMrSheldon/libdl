#include <dl/tensor/math.hpp>
#include <dl/tensor/tensor.hpp>

#include <dl/device.hpp>

using namespace dl;

TensorPtr dl::pow(TensorPtr base, float exponent) noexcept {
	/** \todo add support for autograd **/
	return base->pow(exponent);
}

TensorPtr dl::mean(TensorPtr x) noexcept {
	/** \todo add support for autograd **/
	/*if (x->requiresGrad()) {
		auto size = 1.0f / x->shape(0);
		x->gradfn = [size]() { return size; };
	}
	return x->mean();*/
	auto tensor = x->mean();
	if (tensor->requiresGrad()) {
		auto size = (float)x->shape(0);
		tensor->gradfn = [x, size](TensorPtr ptr) {
			/** \todo fix **/
			if (x->grad == nullptr)
				x->grad = dl::ones_like(x);
			if (x->grad)
				x->grad = x->grad / size;
			else
				x->grad = dl::ones_like(x) / size;
			x->gradfn(x->grad);
		};
	}
	return tensor;
}

// TensorPtr relu(TensorPtr x) noexcept { return max(x, 0); }

TensorPtr dl::operator+(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	if (left->requiresGrad())
		left->gradfn = [left](TensorPtr ptr) { dl::constant(1.0, left->device()); };
	if (right->requiresGrad())
		right->gradfn = [right](TensorPtr ptr) { dl::constant(1.0, right->device()); };
	return left->add(right);
}
TensorPtr dl::operator-(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	left->gradfn = [left](TensorPtr ptr) { dl::constant(1.0, left->device()); };
	right->gradfn = [right](TensorPtr ptr) { dl::constant(-1.0, right->device()); };
	return left->sub(right);
}
TensorPtr dl::operator*(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	auto tensor = left->mul(right);
	if (tensor->requiresGrad()) {
		tensor->gradfn = [left, right](TensorPtr ptr) {
			auto lgrad = right->clone();
			lgrad->discardGradient();
			auto rgrad = left->clone();
			rgrad->discardGradient();

			left->grad = lgrad * ptr;
			right->grad = rgrad * ptr;
		};
	}
	return tensor;
}
TensorPtr dl::operator/(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	return left->div(right);
}

TensorPtr dl::matmul(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	return left->matmul(right);
}

std::ostream& dl::operator<<(std::ostream& stream, TensorPtr tensor) noexcept { return tensor->writeToStream(stream); }

bool dl::operator==(dl::TensorPtr left, dl::TensorPtr right) noexcept {
	if (left->shape() != right->shape())
		return false;
	throw std::runtime_error("Not Implemented");
	/*auto flatleft = left->flatten();
	auto flatright = right->flatten();
	for (auto it1 = std::begin(flatleft), it2 = std::begin(flatright);
		 it1 != std::end(flatleft) && it2 != std::end(flatright); ++it1, ++it2)
		if ((double)*it1 != (double)*it2)
			return false;
	return true;*/
}