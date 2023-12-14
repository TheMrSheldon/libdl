add_executable(example_01_simple
	main.cpp
)


target_compile_features(example_01_simple PUBLIC cxx_std_23)
# target_include_directories(example_01_simple PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(example_01_simple dl)