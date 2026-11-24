cmake_minimum_required(VERSION 3.20)
project(mv_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_library(
    mv_cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/mv_cuda.cu
)


set_target_properties(mv_cuda PROPERTIES CUDA_ARCHITECTURES "75;86")
target_include_directories(mv_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(mv_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mv_cuda PRIVATE glm)

