cmake_minimum_required(VERSION 3.10)
project(pnk_svm)

option(USE_CUDA_BACKEND "Option description" ON)
if (USE_CUDA_BACKEND)
  add_definitions(-DUSE_CUDA_BACKEND)
endif()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(process_data process_data.cpp)
target_link_libraries(process_data ${OpenCV_LIBS})

#SET(CMAKE_CXX_FLAGS  "-Ofast -fopenmp")
#SET(CMAKE_C_FLAGS  "-Ofast -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin g++ -O3 -m64
  -gencode arch=compute_52,code=sm_52
  -gencode arch=compute_60,code=sm_60
  -gencode arch=compute_61,code=sm_61
  -gencode arch=compute_70,code=sm_70
  -gencode arch=compute_75,code=sm_75
  -gencode arch=compute_80,code=sm_80
  -gencode arch=compute_86,code=sm_86
  -gencode arch=compute_86,code=sm_86
  --extended-lambda" )
cuda_add_executable(training training.cpp configuration.cpp svm.cu dataset.cpp)
target_link_libraries(training cudart cublas)
cuda_add_executable(evaluate evaluate.cpp configuration.cpp svm.cu dataset.cpp)
target_link_libraries(evaluate cudart cublas)
