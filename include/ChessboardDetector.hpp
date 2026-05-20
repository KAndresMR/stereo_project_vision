#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "CalibrationConfig.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// DetectionResult
//
// Plain data struct returned by ChessboardDetector::detect().
// Keeping it separate from the detector makes it easy to pass around,
// store for later, and inspect in tests.
// ─────────────────────────────────────────────────────────────────────────────
struct DetectionResult {
    bool found = false;

    // Sub-pixel refined corner positions in image coordinates.
    // Size == boardSize.width * boardSize.height when found == true.
    std::vector<cv::Point2f> corners;
};

// ─────────────────────────────────────────────────────────────────────────────
// ChessboardDetector
//
// Responsibilities:
//   1. Run findChessboardCorners on a grayscale frame.
//   2. If found, refine corners to sub-pixel accuracy with cornerSubPix.
//   3. Draw corners onto a colour display frame (caller provides it).
//
// Thread safety: detect() is stateless after construction — safe to call
// from multiple threads with different frames if needed in the future.
// ─────────────────────────────────────────────────────────────────────────────
class ChessboardDetector {
public:
    explicit ChessboardDetector(const CalibrationConfig& config);

    // grayFrame   — CV_8UC1 grayscale input (not modified).
    // displayFrame — CV_8UC3 colour frame; corners are drawn onto it if found.
    // Returns DetectionResult with found flag and refined corner positions.
    DetectionResult detect(const cv::Mat& grayFrame,
                           cv::Mat&       displayFrame) const;

private:
    CalibrationConfig config_;
};
