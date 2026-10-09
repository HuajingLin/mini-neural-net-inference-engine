cmake_minimum_required(VERSION 3.16)
project(mini_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(mini INTERFACE)
target_include_directories(mini INTERFACE include)

enable_testing()
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE mini)
add_test(NAME tensor COMMAND test_tensor)