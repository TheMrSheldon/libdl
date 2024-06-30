add_executable(example_04_mnist
	main.cpp
)

target_compile_features(example_04_mnist PUBLIC cxx_std_23)
target_link_libraries(example_04_mnist dl)

##########################################################################################
# Libraries
##########################################################################################
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ENABLE_CMAKE ON)
find_package(Boost 1.45.0 COMPONENTS iostreams REQUIRED)
target_link_libraries(example_04_mnist Boost::iostreams)