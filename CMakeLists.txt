cmake_minimum_required(VERSION 3.15)
project(lane_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(lane_detection lane_detection.cpp)
target_link_libraries(lane_detection PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(lane_detection PRIVATE /W4 /utf-8)
endif()
