cmake_minimum_required(VERSION 3.20)
project(fill_with_order LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_library(
    fill_with_order
    ${CMAKE_CURRENT_SOURCE_DIR}/fill_with_order.cu
)


set_target_properties(fill_with_order PROPERTIES CUDA_ARCHITECTURES "75;86")
target_include_directories(fill_with_order PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(fill_with_order PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

