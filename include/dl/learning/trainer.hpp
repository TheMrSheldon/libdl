#pragma once

namespace dl {
	class Model;
	class Device;

	class Trainer final {
	public:
		struct Settings {
			bool enableCheckpointing = false;
		};

	private:
		Settings settings;

		Trainer(const Trainer& other) = delete;
		Trainer(Trainer&& other) = delete;
		Trainer& operator=(const Trainer& other) = delete;
		Trainer& operator=(Trainer&& other) = delete;

	public:
		Trainer(Settings settings = {});

		void fit(const Model& model) const noexcept;
		void validate(const Model& model) const noexcept;
		void test(const Model& model) const noexcept;
	};
} // namespace dl