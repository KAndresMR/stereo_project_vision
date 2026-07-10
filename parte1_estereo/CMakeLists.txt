cmake_minimum_required(VERSION 3.16)
project(StereoVision)

set(CMAKE_CXX_STANDARD 17)
# AQUÍ FUERZAS EL MODO DE MÁXIMO RENDIMIENTO (C++ Optimizado al 100%)
set(CMAKE_BUILD_TYPE Release)

# Apple Silicon Homebrew
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

file(GLOB SRC_FILES
    src/*.cpp
)

add_executable(
    StereoVision
    ${SRC_FILES}
)

target_link_libraries(
    StereoVision
    ${OpenCV_LIBS}
    CURL::libcurl
)
