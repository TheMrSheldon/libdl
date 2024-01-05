# libdl

# Usage
## CMake
```cmake
FetchContent_Declare(libdl GIT_REPOSITORY https://github.com/TheMrSheldon/libdl.git)
FetchContent_MakeAvailable(libdl)
target_link_libraries(<mytarget> PUBLIC libdl)
```