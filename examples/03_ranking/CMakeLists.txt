add_executable(example_03_ranking
	main.cpp
)


target_compile_features(example_03_ranking PUBLIC cxx_std_23)
target_link_libraries(example_03_ranking ir)