cmake_minimum_required(VERSION 3.1)
project(opencv-test)

set(OpenCV_DIR "D:\\Libraries\\OpenCV\\opencv\\build\\x64\\vc16\\lib")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} opencv-test.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})