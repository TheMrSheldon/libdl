add_executable(example_06_ranking
	main.cpp
)


target_compile_features(example_06_ranking PUBLIC cxx_std_23)
target_link_libraries(example_06_ranking ir)