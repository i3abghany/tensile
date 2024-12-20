cmake_minimum_required(VERSION 3.21)
project(tensile)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

include_directories(include)

add_compile_options(-Wall
    -Wextra
    -Wpedantic
    -Werror
    -march=znver2
    -fopenmp
)

add_link_options(-fopenmp)

add_executable(tensile
    src/main.cpp
    src/index_parser.cpp
    src/logger.cpp
    src/unimpl.cpp
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(tensile PRIVATE TENSILE_LOGGING_ENABLED)
endif()

enable_testing()
add_subdirectory(test)