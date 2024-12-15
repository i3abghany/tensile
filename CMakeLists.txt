cmake_minimum_required(VERSION 3.21)
project(tensile)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

include_directories(include)

add_compile_options(-Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-unused-but-set-variable
    -Wno-unused-const-variable
)

add_executable(tensile
    src/main.cpp
    src/index_parser.cpp
    src/logger.cpp
)

enable_testing()

add_executable(tests
    src/index_parser.cpp

    test/init_tests.cpp
    test/slicing_tests.cpp
    test/index_parser_tests.cpp
    test/expand_tests.cpp
    test/test_utils.cpp
    test/shape_tests.cpp
    test/add_tests.cpp
)

target_link_libraries(tests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)
