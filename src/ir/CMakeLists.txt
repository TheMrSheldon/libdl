include(FetchContent)

# Library
add_library(ir STATIC
    data/datasets.cpp
)
target_compile_features(ir PUBLIC cxx_std_23)
target_include_directories(ir PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../include)
target_link_libraries(ir PUBLIC dl)

##########################################################################################
# Libraries
##########################################################################################
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ENABLE_CMAKE ON)
find_package(Boost 1.45.0 COMPONENTS iostreams REQUIRED)
target_link_libraries(ir PUBLIC Boost::iostreams)