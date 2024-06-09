#include <dl/device.hpp>
#include <dl/io/weightsfile.hpp>
#include <dl/model/model.hpp>

#include <sstream>

#include <catch2/catch_test_macros.hpp>

class DummyModel : public dl::Model<void(void)> {
public:
	dl::TensorPtr weight;
	DummyModel() noexcept : weight(dl::empty({20, 20})) { registerParameter("weight", weight); }

	virtual void forward() {}
};

TEST_CASE("Safetensors", "[IO]") {
	DummyModel linear;
	linear.weight = dl::rand_like(linear.weight);

	std::stringstream stream;
	dl::io::safetensorsFormat.writeModelToStream(linear, stream);

	stream.seekg(0, std::ios::beg);

	DummyModel loaded;
	REQUIRE(dl::io::safetensorsFormat.loadModelFromStream(loaded, stream));
	REQUIRE(dl::allclose(linear.weight, loaded.weight));
}

TEST_CASE("GGUF", "[IO]") {
	DummyModel linear;
	linear.weight = dl::rand_like(linear.weight);

	std::stringstream stream;
	dl::io::ggufFormat.writeModelToStream(linear, stream);

	stream.seekg(0, std::ios::beg);

	DummyModel loaded;
	REQUIRE(dl::io::ggufFormat.loadModelFromStream(loaded, stream));
	REQUIRE(dl::allclose(linear.weight, loaded.weight));
}