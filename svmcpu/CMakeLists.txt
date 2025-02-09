cmake_minimum_required(VERSION 3.10)
project(pnk_svm)

option(USE_CUDA_BACKEND "Option description" OFF)
if (USE_CUDA_BACKEND)
  add_definitions(-DUSE_CUDA_BACKEND)
endif()

find_package(OpenCV REQUIRED)

add_executable(process_data process_data.cpp)
target_link_libraries(process_data ${OpenCV_LIBS})

SET(CMAKE_CXX_FLAGS  "-Ofast -fopenmp")
SET(CMAKE_C_FLAGS  "-Ofast -fopenmp")
add_executable(training training.cpp svm.cpp dataset.cpp)
add_executable(evaluate evaluate.cpp svm.cpp dataset.cpp)
