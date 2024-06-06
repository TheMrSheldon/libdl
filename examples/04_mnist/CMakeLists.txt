add_executable(example_04_mnist
	main.cpp
)

target_compile_features(example_04_mnist PUBLIC cxx_std_23)
target_link_libraries(example_04_mnist dl)