#pragma once
#include <opencv2/opencv.hpp>

namespace DebugViews {

inline cv::Mat colorizeDisparity(
    const cv::Mat& disparity,
    int numDisparities,
    int minDisparity = 0)
{
    cv::Mat disp32f;
    disparity.convertTo(disp32f, CV_32F, 1.0 / 16.0);

    cv::threshold(
        disp32f,
        disp32f,
        0.0,
        0.0,
        cv::THRESH_TOZERO);

    cv::Mat disp8u;
    disp32f.convertTo(
        disp8u,
        CV_8U,
        255.0 / numDisparities);

    cv::Mat colored;
    cv::applyColorMap(
        disp8u,
        colored,
        cv::COLORMAP_TURBO);

    cv::Mat invalid =
        (disparity <= minDisparity * 16);

    colored.setTo(cv::Scalar(0,0,0), invalid);

    return colored;
}

inline cv::Mat stackHorizontal(
    const std::vector<cv::Mat>& imgs)
{
    cv::Mat out;
    cv::hconcat(imgs, out);
    return out;
}

inline void label(
    cv::Mat& img,
    const std::string& text)
{
    cv::putText(
        img,
        text,
        {10, 25},
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(255,255,255),
        2);
}

}