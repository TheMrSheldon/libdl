#include <dl/tensor/math.hpp>
#include <dl/tensor/tensorimpl.hpp>

#include <dl/device.hpp>

using namespace dl;

Tensor dl::pow(Tensor& base, float exponent) noexcept {
	/** \todo add support for autograd **/
	return base->pow(exponent);
}

Tensor dl::pow(Tensor&& base, float exponent) noexcept {
	/** \todo add support for autograd **/
	return base->pow(exponent);
}

Tensor dl::mean(Tensor& x) noexcept {
	/** \todo add support for autograd **/
	auto tensor = x->mean();
	if (tensor->requiresGrad()) {
		auto size = (float)x->shape(0);
		tensor->gradfn = [x, size](Tensor& ptr) mutable {
			if (x->grad == nullptr)
				x->grad = std::move(ptr * dl::ones_like(x) / size);
			else
				x->grad = std::move(x->grad + (ptr * dl::ones_like(x) / size));
			if (x->gradfn)
				x->gradfn(x->grad);
			else
				assert(x->requiresGrad());
		};
	}
	return tensor;
}

Tensor dl::mean(Tensor&& x) noexcept {
	auto tensor = x->mean();
	if (tensor->requiresGrad()) {
		auto size = (float)x->shape(0);
		tensor->gradfn = [copy = std::move(x), size](Tensor& ptr) mutable {
			if (copy->grad == nullptr)
				copy->grad = std::move(ptr * dl::ones_like(copy) / size);
			else
				copy->grad = std::move(copy->grad + (ptr * dl::ones_like(copy) / size));
			if (copy->gradfn)
				copy->gradfn(copy->grad);
			else
				assert(copy->requiresGrad());
		};
	}
	return tensor;
}

// Tensor relu(Tensor x) noexcept { return max(x, 0); }

Tensor dl::operator+(Tensor& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	if (left->requiresGrad())
		left->gradfn = [left](Tensor& ptr) { dl::constant(1.0, left->device()); };
	if (right->requiresGrad())
		right->gradfn = [right](Tensor& ptr) { dl::constant(1.0, right->device()); };
	return left->add(right);
}
Tensor dl::operator-(Tensor& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	left->gradfn = [left](Tensor& ptr) { dl::constant(1.0, left->device()); };
	right->gradfn = [right](Tensor& ptr) { dl::constant(-1.0, right->device()); };
	return left->sub(right);
}
Tensor dl::operator*(Tensor& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	auto tensor = left->mul(right);
	if (tensor->requiresGrad()) {
		tensor->gradfn = [&left, &right](Tensor& ptr) {
			auto lgrad = right;
			lgrad->discardGradient();
			auto rgrad = left;
			rgrad->discardGradient();

			left->grad = lgrad * ptr;
			right->grad = rgrad * ptr;
		};
	}
	return tensor;
}
Tensor dl::operator/(Tensor& left, Tensor right) noexcept {
	/** \todo add support for autograd **/
	return left->div(right);
}
Tensor dl::operator/(Tensor& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	return left->div(right);
}

Tensor dl::matmul(Tensor& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	return left->matmul(right);
}
Tensor dl::matmul(Tensor&& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	return left->matmul(right);
}

std::ostream& dl::operator<<(std::ostream& stream, const Tensor& tensor) noexcept {
	if (tensor == nullptr)
		return stream << "null";
	return tensor->writeToStream(stream);
}

bool dl::operator==(const dl::Tensor& left, const dl::Tensor& right) noexcept {
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