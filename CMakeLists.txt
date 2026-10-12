cmake_minimum_required(VERSION 3.16)
project(mini_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE) 
set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(mini STATIC src/ops.cpp)
target_include_directories(mini PUBLIC include)

enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE mini)
add_test(NAME tensor COMMAND test_tensor)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE mini)
add_test(NAME gemm COMMAND test_gemm ${CMAKE_SOURCE_DIR}/tests/gemm_ref.txt)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE mini)
add_test(NAME ops COMMAND test_ops ${CMAKE_SOURCE_DIR}/tests/ops_ref.txt)
