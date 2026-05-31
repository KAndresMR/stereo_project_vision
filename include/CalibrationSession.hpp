#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "CalibrationConfig.hpp"

class CalibrationSession {
public:

    explicit CalibrationSession(
        const CalibrationConfig& config);

    bool savePair(
        const cv::Mat& left,
        const cv::Mat& right);

    int pairCount() const {
        return pairCount_;
    }

    bool isComplete() const {
        return pairCount_ >= config_.targetPairs;
    }

private:

    CalibrationConfig config_;

    int pairCount_ = 0;

    void ensureDirectories() const;

    void scanExistingDataset();

    std::string leftPath(int n) const;
    std::string rightPath(int n) const;
};