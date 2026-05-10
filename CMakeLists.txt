cmake_minimum_required(VERSION 3.16)
project(StereoVision)

set(CMAKE_CXX_STANDARD 17)

# En M1 Mac, Homebrew instala en /opt/homebrew (no /usr/local como en Intel)
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

add_executable(StereoVision main.cpp)

target_include_directories(StereoVision PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(StereoVision PRIVATE
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
)
