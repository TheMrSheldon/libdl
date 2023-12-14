#include <dl/devices/cpu.hpp>

using Shape = dl::Shape;
using TensorPtr = dl::TensorPtr;

using CPUDevice = dl::cpu::CPUDevice;

TensorPtr CPUDevice::emptyTensor(Shape shape) noexcept {
	
}