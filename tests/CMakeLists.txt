include(CTest)
include(FetchContent)

FetchContent_Declare(
	catch
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG v3.3.2
)
FetchContent_MakeAvailable(catch)

add_executable(tests
	# Add test files here
	model/transformer/wordpiece.cpp
	tensor/autodiff.cpp
	tensor/basics.cpp
)

# file(COPY testfiles ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain dl)
target_compile_features(tests PRIVATE cxx_std_23)

add_test(NAME dl_test COMMAND tests
	#WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/testfiles
)
