cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

option(LIBDL_BUILD_TESTS "Build tests" ON)
option(LIBDL_BUILD_DOCS "Build documentation" ON)
option(LIBDL_BUILD_DEB "Build debian package" ON)
option(LIBDL_ONLY_DOCS "Build only documentation -- this disables tests and others" OFF)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/dl/dl.hpp" LIBDL_VERSION_MACRO REGEX "^\tconstexpr const auto version = \"(([0-9]+\\.)*[0-9]+)\";$")
string(REGEX REPLACE "^\tconstexpr const auto version = \"(([0-9]+\\.)*[0-9]+)\"\\;$" "\\1" LIBDL_VERSION ${LIBDL_VERSION_MACRO})

project(libdl VERSION ${LIBDL_VERSION} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT LIBDL_ONLY_DOCS)
	add_subdirectory(src)
endif()

########################################################################################################################
# Testing                                                                                                              #
########################################################################################################################
# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if (NOT LIBDL_ONLY_DOCS AND (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING) AND LIBDL_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()


########################################################################################################################
# Examples                                                                                                             #
########################################################################################################################
if (NOT LIBDL_ONLY_DOCS)
add_subdirectory(examples)
endif()

##########################################################################################
# Documentation
##########################################################################################
# Documentation only available if this is the main app
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME) AND LIBDL_BUILD_DOCS)
	find_package(Doxygen REQUIRED)
	include(FetchContent)

	FetchContent_Declare(cppreftags
		URL https://upload.cppreference.com/mwiki/images/f/f8/cppreference-doxygen-web.tag.xml
		DOWNLOAD_NO_EXTRACT TRUE
	)
	FetchContent_MakeAvailable(cppreftags)

	## Set Doxygen configuration options (THIS REPLACES THE DOXYFILE!)
	set(DOXYGEN_PROJECT_BRIEF "Simple yet powerful deep learning")
	set(DOXYGEN_PROJECT_LOGO "thirdparty/twemoji-brain.png")
	set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/documentation)
	set(DOXYGEN_STRIP_FROM_PATH "include")
	set(DOXYGEN_EXTRACT_PRIVATE "NO")
	set(DOXYGEN_EXTRACT_STATIC "YES")
	set(DOXYGEN_USE_MATHJAX "YES")
	set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
	set(DOXYGEN_SOURCE_BROWSER "YES")
	set(DOXYGEN_INLINE_SOURCES "YES")
	set(DOXYGEN_IMAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/docs/media)
	set(DOXYGEN_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/examples)
	set(DOXYGEN_CITE_BIB_FILES ${CMAKE_CURRENT_SOURCE_DIR}/docs/references.bib)
	set(DOXYGEN_TAGFILES "${cppreftags_SOURCE_DIR}/cppreference-doxygen-web.tag.xml=http://en.cppreference.com/w/")

	set(DOXYGEN_GENERATE_HTML "YES")

    ## Install and configure the "Doxygen Awesome" theme
	FetchContent_Declare(doxawesome GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css GIT_TAG v2.3.2)
	FetchContent_MakeAvailable(doxawesome)
	set(DOXYGEN_GENERATE_TREEVIEW "YES")
	set(DOXYGEN_DISABLE_INDEX "NO")
	set(DOXYGEN_FULL_SIDEBAR "NO")
	set(DOXYGEN_HTML_EXTRA_STYLESHEET ${doxawesome_SOURCE_DIR}/doxygen-awesome.css ${doxawesome_SOURCE_DIR}/doxygen-awesome-sidebar-only.css ${doxawesome_SOURCE_DIR}/doxygen-awesome-sidebar-only-darkmode-toggle.css ${doxawesome_SOURCE_DIR}/doxygen-awesome-tabs.js ${doxawesome_SOURCE_DIR}/doxygen-custom/custom.css)
	set(DOXYGEN_HTML_COLORSTYLE "LIGHT")
	set(DOXYGEN_HTML_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-header.html)
	set(DOXYGEN_HTML_EXTRA_FILES ${doxawesome_SOURCE_DIR}/doxygen-awesome-darkmode-toggle.js ${doxawesome_SOURCE_DIR}/doxygen-awesome-fragment-copy-button.js ${doxawesome_SOURCE_DIR}/doxygen-awesome-interactive-toc.js ${doxawesome_SOURCE_DIR}/doxygen-awesome-paragraph-link.js)

	## Build Documentation
	doxygen_add_docs(
		libdl_docs
		${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${CMAKE_CURRENT_SOURCE_DIR}/docs
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)

	# Copy Files over that Doxygen has not found (hopefully Doxygen will do this automatically in the future)
	add_custom_command(TARGET libdl_docs POST_BUILD
    	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/twemoji-brain.svg ${DOXYGEN_OUTPUT_DIRECTORY}/html/thirdparty/twemoji-brain.svg
	)
endif()

##########################################################################################
# Debian Package
##########################################################################################
# Debian Packaging only available if this is the main app
if(NOT LIBDL_ONLY_DOCS AND (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME) AND LIBDL_BUILD_DEB)
	install(TARGETS dl ARCHIVE DESTINATION lib COMPONENT libdl)
	install(DIRECTORY include/dl DESTINATION include COMPONENT libdl)
	install(TARGETS ir ARCHIVE DESTINATION lib COMPONENT libir)
	install(DIRECTORY include/ir DESTINATION include COMPONENT libir)
	install(DIRECTORY include/nlp DESTINATION include COMPONENT libnlp)
	cpack_add_component(libdl)
	cpack_add_component(libir)
	cpack_add_component(libnlp)

	SET(CPACK_GENERATOR "DEB")
	SET(CPACK_PACKAGE_NAME "libdl-full-dev")
	SET(CPACK_SET_DESTDIR TRUE)
	SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "not-set")
	SET(CPACK_PACKAGE_VERSION "0.0.1rc")
	include(GNUInstallDirs)
	SET(CPACK_DEBIAN_PACKAGE_DESCRIPTION "A simple yet powerful deep learning library")
	SET(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
	SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
	SET(CPACK_DEBIAN_PACKAGE_DEPENDS "")
	SET(CPACK_PACKAGE_VENDOR "")
	include(CPack)
endif()