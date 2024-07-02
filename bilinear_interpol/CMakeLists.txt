cmake_minimum_required(VERSION 3.10)

project(bilinear)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
enable_language(CUDA)

#cuda env
include_directories(/usr/local/cuda-11.7/include/)
link_directories(/usr/local/cuda-11.7/lib64/)

# opencv env
find_package(OpenCV REQUIRED)
include_directories(${Opencv_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cpp ${SRCS})
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})