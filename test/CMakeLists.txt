include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(tensile_tests
    ../src/index_parser.cpp
    ../src/logger.cpp
    ../src/unimpl.cpp

    init_tests.cpp
    slicing_tests.cpp
    index_parser_tests.cpp
    expand_tests.cpp
    test_utils.cpp
    shape_tests.cpp
    add_tests.cpp
    matmul2d_tests.cpp
    matmul3d_tests.cpp
)

target_link_libraries(tensile_tests PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tensile_tests)