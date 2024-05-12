include(FetchContent)

# Library
add_library(dl STATIC
    logging.cpp
    device.cpp
    devices/cpu.cpp
    learning/observers/earlystopping.cpp
    learning/observers/ncurses.cpp
    model/model.cpp
    model/layernorm.cpp
    model/linear.cpp
    model/transformer/transformer.cpp
    model/transformer/wordpiece.cpp
    tensor/math.cpp
    tensor/tensor.cpp
    tensor/tensorimpl.cpp

    utils/urlstream.cpp
)
target_compile_features(dl PUBLIC cxx_std_23)
target_include_directories(dl PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../include)

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
target_compile_definitions(dl PRIVATE "-DXTENSOR_USE_XSIMD")


##########################################################################################
# Libraries
##########################################################################################
# xtensor
set(BUILD_TESTS OFF)
set(XTENSOR_USE_XSIMD ON)
set(XTENSOR_USE_OPENMP ON)

# Dependency of xtensor
FetchContent_Declare(xtl GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git GIT_TAG 0.7.5)
FetchContent_MakeAvailable(xtl)
# Dependency of xtensor
FetchContent_Declare(xsimd GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git GIT_TAG 10.0.0)
FetchContent_MakeAvailable(xsimd)

FetchContent_Declare(xtensor GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git GIT_TAG 0.24.7)
FetchContent_MakeAvailable(xtensor)

# Dependency of xtensor-blas
FetchContent_Declare(openblas GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git GIT_TAG v0.3.25)
FetchContent_MakeAvailable(openblas)

# Dependency of xtensor
FetchContent_Declare(xblas GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git GIT_TAG 0.20.0)
FetchContent_MakeAvailable(xblas)
target_link_libraries(dl PRIVATE xtensor xsimd xtensor-blas openblas)


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
target_link_libraries(dl PRIVATE ryml::ryml)

# libcpr
FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
                         GIT_TAG 1.10.5)
FetchContent_MakeAvailable(cpr)
target_link_libraries(dl PRIVATE cpr::cpr)

# hat-trie (for WordPiece tokenization)
FetchContent_Declare(hattrie GIT_REPOSITORY https://github.com/Tessil/hat-trie.git GIT_TAG 906e6ab)
FetchContent_MakeAvailable(hattrie)
target_link_libraries(dl PRIVATE tsl::hat_trie)

# TODO: make optional since this is only necessary for a pretty UI during training
# ncurses
set(CURSES_NEED_NCURSES TRUE)
find_package(Curses REQUIRED)
target_link_libraries(dl PRIVATE ${CURSES_LIBRARIES})
target_include_directories(dl PRIVATE ${CURSES_INCLUDE_DIRS})

# ftxui
FetchContent_Declare(ftxui GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git GIT_TAG v5.0.0)
FetchContent_MakeAvailable(ftxui)
target_link_libraries(dl PRIVATE ftxui::screen ftxui::component ftxui::dom)