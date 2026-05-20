#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "CalibrationConfig.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// DetectionResult — everything the detector knows about one attempt.
// ─────────────────────────────────────────────────────────────────────────────
struct DetectionResult {
    bool   found       = false;
    bool   claheUsed   = false;   // was CLAHE applied before detection?
    double brightness  = 0.0;     // mean pixel value of the grayscale frame (0–255)
    double contrast    = 0.0;     // Michelson contrast of the frame (0–1)
    int    cornersFound    = 0;   // how many corners were found (even if incomplete)
    int    cornersExpected = 0;   // boardSize.width * boardSize.height

    std::vector<cv::Point2f> corners;  // refined corners (only valid when found==true)
};

// ─────────────────────────────────────────────────────────────────────────────
// ChessboardDetector
//
// Pipeline per call:
//   1. Measure brightness — decide whether CLAHE is needed
//   2. Optionally apply CLAHE to a working copy of the gray frame
//   3. findChessboardCorners (FAST_CHECK rejects quickly when board absent)
//   4. cornerSubPix — sub-pixel refinement (only when found)
//   5. drawChessboardCorners on the colour display frame (only when found)
// ─────────────────────────────────────────────────────────────────────────────
class ChessboardDetector {
public:
    explicit ChessboardDetector(const CalibrationConfig& config);

    // grayFrame    — CV_8UC1 grayscale (not modified)
    // displayFrame — CV_8UC3 colour frame; corners drawn on it when found
    DetectionResult detect(const cv::Mat& grayFrame,
                           cv::Mat&       displayFrame) const;

private:
    CalibrationConfig config_;

    double measureBrightness(const cv::Mat& gray) const;
    double measureContrast(const cv::Mat& gray) const;
    cv::Mat applyClahe(const cv::Mat& gray) const;
};