#include <dl/device.hpp>
#include <dl/io/weightsfile.hpp>
#include <dl/model/linear.hpp>

#include <sstream>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Safetensors", "[IO]") {
	auto linear = dl::Linear(10, 20);
	dl::TensorPtr& weight = linear.parameters().find("weight")->second;
	weight = dl::rand_like(weight);

	std::stringstream stream;
	dl::io::safetensorsFormat.writeModelToStream(linear, stream);

	stream.seekg(0, std::ios::beg);

	auto loaded = dl::Linear(10, 20);
	dl::TensorPtr& loadedWeight = loaded.parameters().find("weight")->second;
	REQUIRE(dl::io::safetensorsFormat.loadModelFromStream(loaded, stream));
	REQUIRE(dl::allclose(weight, loadedWeight));
}

TEST_CASE("GGUF", "[IO]") {
	auto linear = dl::Linear(10, 20);
	dl::TensorPtr& weight = linear.parameters().find("weight")->second;
	weight = dl::rand_like(weight);

	std::stringstream stream;
	dl::io::ggufFormat.writeModelToStream(linear, stream);

	stream.seekg(0, std::ios::beg);

	auto loaded = dl::Linear(10, 20);
	dl::TensorPtr& loadedWeight = loaded.parameters().find("weight")->second;
	REQUIRE(dl::io::ggufFormat.loadModelFromStream(loaded, stream));
	REQUIRE(dl::allclose(weight, loadedWeight));
}