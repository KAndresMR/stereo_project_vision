#pragma once
#include <opencv2/opencv.hpp>
#include "CalibrationConfig.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// CalibrationSession
//
// Manages one calibration capture session:
//   - Creates and owns the output directory structure.
//   - Counts captured pairs and enforces the target.
//   - Saves stereo pairs as JPEG with zero-padded filenames.
//
// NOT responsible for detection or display — those live in other classes.
// ─────────────────────────────────────────────────────────────────────────────
class CalibrationSession {
public:
    explicit CalibrationSession(const CalibrationConfig& config);

    // Save a left/right pair. Returns true on success.
    // Filenames: left/left_01.jpg, right/right_01.jpg, etc.
    bool savePair(const cv::Mat& left, const cv::Mat& right);

    int  pairCount()  const { return pairCount_; }
    bool isComplete() const { return pairCount_ >= config_.targetPairs; }

private:
    CalibrationConfig config_;
    int pairCount_ = 0;

    void        ensureDirectories() const;
    std::string leftPath(int n)    const;
    std::string rightPath(int n)   const;
};
