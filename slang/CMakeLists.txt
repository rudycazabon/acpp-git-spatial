cmake_minimum_required(VERSION 3.24)
project(SlangGisCompute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SLANG_SEARCH_PATHS
    /usr/local
    /opt/slang
    /home/rudycazabon/Libraries/vulkan-sdk/1.4.350.1/x86_64
    $ENV{SLANG_DIR}
)

find_path(SLANG_INCLUDE_DIR slang.h PATHS ${SLANG_SEARCH_PATHS} PATH_SUFFIXES include/slang slang)

# Compiler/reflection API
find_library(SLANG_LIBRARY NAMES slang PATHS ${SLANG_SEARCH_PATHS} PATH_SUFFIXES lib lib64 bin)

# Graphics wrapper API -- gfxCreateDevice and friends live here, NOT in libslang.so
find_library(SLANG_GFX_LIBRARY NAMES gfx PATHS ${SLANG_SEARCH_PATHS} PATH_SUFFIXES lib lib64 bin)

if(NOT SLANG_INCLUDE_DIR OR NOT SLANG_LIBRARY OR NOT SLANG_GFX_LIBRARY)
    message(FATAL_ERROR "Slang SDK not fully found. INCLUDE_DIR=${SLANG_INCLUDE_DIR} SLANG_LIBRARY=${SLANG_LIBRARY} SLANG_GFX_LIBRARY=${SLANG_GFX_LIBRARY}")
endif()

add_executable(SlangGisCompute main.cpp)

target_include_directories(SlangGisCompute PRIVATE ${SLANG_INCLUDE_DIR})
target_link_libraries(SlangGisCompute PRIVATE ${SLANG_LIBRARY} ${SLANG_GFX_LIBRARY})

add_custom_command(TARGET SlangGisCompute POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/spatial_transform.slang
    $<TARGET_FILE_DIR:SlangGisCompute>/spatial_transform.slang
)