# ==============================================================================
# Configuración de compilación para la aplicación de escritorio TriggerApp (C++)
# Detección de vehículos en tiempo real mediante HOG + SVM y OpenCV
# ==============================================================================
cmake_minimum_required(VERSION 3.16)
project(TriggerApp)

set(CMAKE_CXX_STANDARD 17)
# Fundamental para procesamiento de imágenes (HOG+SVM) en tiempo real
set(CMAKE_BUILD_TYPE Release) 

# Apple Silicon Homebrew
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    TriggerApp
    src/main.cpp
)

target_link_libraries(
    TriggerApp
    ${OpenCV_LIBS}
    CURL::libcurl
)
