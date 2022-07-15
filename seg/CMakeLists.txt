cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(TensorRTSample)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

# TARGETS --------------------------------------------------------------------------------------------------------------
set(SAMPLE_SOURCES
    infer_seg.cpp
)

set(SAMPLES_COMMON_SOURCES
    $<BUILD_INTERFACE:/usr/src/tensorrt/samples/common/logger.cpp>
)

add_executable(infer_seg
    ${SAMPLE_SOURCES}
    ${SAMPLES_COMMON_SOURCES}
)

target_include_directories(infer_seg PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} $<BUILD_INTERFACE:/usr/src/tensorrt/samples/common/>)
target_link_libraries(infer_seg PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})


