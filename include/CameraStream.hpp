#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>

struct CameraStream {
    std::vector<uchar> buffer;

    cv::Mat frame;

    std::mutex frame_mtx;
    std::mutex buffer_mtx;
};