cmake_minimum_required(VERSION 3.20)
project(SyclGeospatialTransform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force AdaptiveCPP-specific compiler configurations if using acpp driver wrapper
# Alternatively, find the AdaptiveCPP package if compiling via standard Clang/GCC
find_package(AdaptiveCPP REQUIRED)

file(GLOB SOURCES "*.cpp")
foreach(SOURCE ${SOURCES})
    get_filename_component(FN ${SOURCE} NAME_WE)
    add_executable(${FN} ${SOURCE})
    add_sycl_to_target(TARGET ${FN})
    target_compile_options(${FN} PRIVATE -O3)
    target_compile_options(${FN} PRIVATE --acpp-targets="omp")
    add_sycl_to_target(TARGET ${FN})
endforeach()

