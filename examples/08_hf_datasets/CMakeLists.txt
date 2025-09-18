add_executable(example_08_hf_datasets
	main.cpp
)

target_compile_features(example_08_hf_datasets PUBLIC cxx_std_23)
target_link_libraries(example_08_hf_datasets dl)

FetchContent_Declare(
	yaml-cpp
	GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
	GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

target_link_libraries(example_08_hf_datasets yaml-cpp::yaml-cpp)