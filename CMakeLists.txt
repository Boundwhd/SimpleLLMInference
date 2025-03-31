cmake_minimum_required(VERSION 3.20)

project(SimpleLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

find_package(OpenBLAS REQUIRED)

include_directories(
    include/base
    include/model
    include/memory
    include/op
    include/kernel
    /usr/local/cuda-11.8/include
)

link_directories(
    /usr/local/cuda-11.8/lib64
)

file(GLOB_RECURSE SOURCES
    "source/memory/*.cpp"
    "source/model/*.cpp"
    "source/op/*.cpp"
    "source/kernel/*.cpp"
)

add_executable(${PROJECT_NAME}
    ${SOURCES}
    demo/main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    cudart
    ${OpenBLAS_LIBRARIES}
)
