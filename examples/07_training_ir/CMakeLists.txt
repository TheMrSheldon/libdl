add_executable(example_07_training_ir
	main.cpp
)


target_compile_features(example_07_training_ir PUBLIC cxx_std_23)
target_link_libraries(example_07_training_ir ir)