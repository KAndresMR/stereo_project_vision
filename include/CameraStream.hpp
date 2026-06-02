#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <chrono>

struct CameraStream {

    std::vector<uchar> buffer;

    cv::Mat frame;

    std::mutex frame_mtx;
    std::mutex buffer_mtx;

    std::chrono::steady_clock::time_point timestamp;

    std::chrono::steady_clock::time_point lastFpsTime =
        std::chrono::steady_clock::now();

    double fps = 0.0;
    int frameCount = 0;
};

inline cv::Mat grabFrame(CameraStream& cam) {
    std::lock_guard<std::mutex> l(cam.frame_mtx);
    return cam.frame.empty() ? cv::Mat{} : cam.frame.clone();
}