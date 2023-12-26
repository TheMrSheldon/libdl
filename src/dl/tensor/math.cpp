#include <dl/tensor/math.hpp>
#include <dl/tensor/tensor.hpp>

using namespace dl;

// TensorPtr relu(TensorPtr x) noexcept { return max(x, 0); }

TensorPtr dl::operator+(TensorPtr left, TensorPtr right) noexcept {
	/** \todo implement **/
	throw std::runtime_error("Not yet implemented");
}
TensorPtr dl::operator-(TensorPtr left, TensorPtr right) noexcept {
	/** \todo implement **/
	throw std::runtime_error("Not yet implemented");
}
TensorPtr dl::operator*(TensorPtr left, TensorPtr right) noexcept {
	/** \todo implement **/
	throw std::runtime_error("Not yet implemented");
}
TensorPtr dl::operator/(TensorPtr left, TensorPtr right) noexcept {
	/** \todo implement **/
	throw std::runtime_error("Not yet implemented");
}

std::ostream& dl::operator<<(std::ostream& stream, TensorPtr tensor) noexcept {
	return tensor->writeToStream(stream);
}