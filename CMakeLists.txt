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
)

add_executable(tensile
    src/main.cpp
    src/index_parser.cpp
    src/logger.cpp
    src/unimpl.cpp
)

enable_testing()
add_subdirectory(test)