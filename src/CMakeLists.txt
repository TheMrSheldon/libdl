include(FetchContent)

# Library
add_library(dl STATIC
    dl/logging.cpp
    dl/devices/cpu.cpp
    dl/model/linear.cpp
    dl/tensor/tensor.cpp
)
target_compile_features(dl PUBLIC cxx_std_23)
target_include_directories(dl PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

# Tell the code about the current git commit hash
# (http://xit0.org/2013/04/cmake-use-git-branch-and-commit-details-in-project/)
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
	COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
target_compile_definitions(dl PRIVATE "-DLIBDL_GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")


##########################################################################################
# Libraries
##########################################################################################
# spdlog for logging
FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git GIT_TAG v1.11.0)
FetchContent_MakeAvailable(spdlog)
target_link_libraries(dl PUBLIC spdlog::spdlog)

# rapidyaml
FetchContent_Declare(ryml
    GIT_REPOSITORY https://github.com/biojppm/rapidyaml.git
    GIT_TAG v0.5.0
    GIT_SHALLOW FALSE  # ensure submodules are checked out
)
FetchContent_MakeAvailable(ryml)
target_link_libraries(dl PUBLIC ryml::ryml)
