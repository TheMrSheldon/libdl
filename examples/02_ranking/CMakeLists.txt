add_executable(example_02_ranking
	main.cpp
)


target_compile_features(example_02_ranking PUBLIC cxx_std_23)
# target_include_directories(example_02_ranking PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(example_02_ranking ir)