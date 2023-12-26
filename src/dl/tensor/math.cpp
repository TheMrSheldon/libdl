#include <dl/tensor/math.hpp>
#include <dl/tensor/tensor.hpp>

using namespace dl;

TensorPtr dl::pow(TensorPtr base, float exponent) noexcept {
	/** \todo add support for autograd **/
	return base->pow(exponent);
}

TensorPtr dl::mean(TensorPtr x) noexcept {
	/** \todo add support for autograd **/
	return x->mean();
}

// TensorPtr relu(TensorPtr x) noexcept { return max(x, 0); }

TensorPtr dl::operator+(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	return left->add(right);
}
TensorPtr dl::operator-(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	return left->sub(right);
}
TensorPtr dl::operator*(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	return left->mul(right);
}
TensorPtr dl::operator/(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	return left->div(right);
}

TensorPtr dl::matmul(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	return left->matmul(right);
}

std::ostream& dl::operator<<(std::ostream& stream, TensorPtr tensor) noexcept {
	return tensor->writeToStream(stream);
}