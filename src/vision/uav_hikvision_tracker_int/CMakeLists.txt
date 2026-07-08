cmake_minimum_required(VERSION 3.14)
project(UAV_Hikvision_Tracker_CPP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_path(RKNN_API_INCLUDE_DIR
    NAMES rknn_api.h
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}
        $ENV{RKNN_API_INCLUDE_DIR}
        /usr/include
        /usr/local/include
)

if(NOT RKNN_API_INCLUDE_DIR)
    message(FATAL_ERROR "rknn_api.h not found. Set RKNN_API_INCLUDE_DIR or install Rockchip rknpu2 headers.")
endif()

include_directories(${RKNN_API_INCLUDE_DIR})
find_library(RKNN_RT_LIB
    NAMES rknnrt librknnrt
    HINTS
        $ENV{RKNN_API_LIB_DIR}
        /usr/lib
        /usr/local/lib
)
if(NOT RKNN_RT_LIB)
    message(WARNING "librknnrt not found in default paths. Set RKNN_API_LIB_DIR if linking fails.")
endif()

add_executable(uav_hik_tracker main.cpp yolo_decoder.cpp bytetrack_lite.cpp)

target_link_libraries(uav_hik_tracker
    ${OpenCV_LIBS}
    ${RKNN_RT_LIB}
    pthread
    rt
)
