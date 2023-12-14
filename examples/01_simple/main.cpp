#include <iostream>

#include <dl/learning/trainer.hpp>
#include <dl/model/model.hpp>
#include <dl/tensor/tensor.hpp>

class MyModel : public dl::Model {
public:
	TensorPtr forward(TensorPtr input) {}
};

int main(int argc, char* argv[]) {
	std::cout << "Hello world" << std::endl;
	dl::Trainer trainer;
	return 0;
}