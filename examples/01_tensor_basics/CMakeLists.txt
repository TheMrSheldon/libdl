add_executable(example_01_tensor_basics
	main.cpp
)


target_compile_features(example_01_tensor_basics PUBLIC cxx_std_23)
target_link_libraries(example_01_tensor_basics libdl::dl)