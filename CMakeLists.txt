cmake_minimum_required(VERSION 3.21)
project(Tensor)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

include_directories(include)

add_executable(Tensor 
    src/main.cpp
)

enable_testing()

add_executable(tests
    test/init_tests.cpp
    test/index_tests.cpp
)

target_link_libraries(tests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)