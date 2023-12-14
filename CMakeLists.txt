cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/dl/dl.hpp" LIBDL_VERSION_MACRO REGEX "^\tconstexpr const auto version = \"(([0-9]+\\.)*[0-9]+)\";$")
string(REGEX REPLACE "^\tconstexpr const auto version = \"(([0-9]+\\.)*[0-9]+)\"\\;$" "\\1" LIBDL_VERSION ${LIBDL_VERSION_MACRO})

project(dl VERSION ${LIBDL_VERSION} LANGUAGES CXX)

option(BUILD_TESTS "Build tests" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

########################################################################################################################
# Testing                                                                                                              #
########################################################################################################################
# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING) AND BUILD_TESTS)
	enable_testing()
	# add_subdirectory(tests)
endif()


########################################################################################################################
# Examples                                                                                                             #
########################################################################################################################
add_subdirectory(examples)