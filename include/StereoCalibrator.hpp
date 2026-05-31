#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "CalibrationConfig.hpp"
#include "ChessboardDetector.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// StereoCalibrator — Phase 2 of the pipeline
//
// Responsibility:
//   Given a stereo image dataset and pre-computed individual intrinsics
//   (from MonoCalibrator), compute the geometric relationship between
//   the two cameras: rotation R, translation T, essential matrix E,
//   and fundamental matrix F.
//
// What it does NOT do:
//   - individual camera calibration  (MonoCalibrator)
//   - rectification                  (Rectifier)
//   - disparity or depth             (SGBMProcessor)
//
// Key design decision: CALIB_FIX_INTRINSIC
//   We fix K_left, dist_left, K_right, dist_right to their pre-computed
//   values and only optimize R and T. This is the correct professional
//   approach — it gives more stable geometry with smaller datasets.
//
// Usage:
//   StereoCalibrator sc(config);
//   auto result = sc.calibrate();
//   if (result.success) sc.saveYAML(result);
// ─────────────────────────────────────────────────────────────────────────────
class StereoCalibrator {
public:

    // ── Result ────────────────────────────────────────────────────────────────
    struct Result {
        bool success = false;

        cv::Mat R;           // 3×3  rotation:    right camera relative to left
        cv::Mat T;           // 3×1  translation: right camera origin in left coords (meters)
        cv::Mat E;           // 3×3  essential matrix
        cv::Mat F;           // 3×3  fundamental matrix

        double rpe        = 0.0;  // RMS stereo reprojection error
        double baselineM  = 0.0;  // |T| in meters (≈ physical separation)
        int    pairsUsed  = 0;
        int    pairsTotal = 0;
        cv::Size imageSize;
    };

    explicit StereoCalibrator(const CalibrationConfig& config);

    // Full offline pipeline: loads intrinsics, detects pairs, runs stereoCalibrate.
    Result calibrate() const;

    // Persist R, T, E, F, Q, R1, R2, P1, P2 to stereo.yaml.
    // Also runs stereoRectify internally to compute and save Q matrix,
    // so everything needed for the next phase is in one file.
    bool saveYAML(const Result& result) const;

    void printSummary(const Result& result) const;

private:
    CalibrationConfig  config_;
    ChessboardDetector detector_;

    // Load K and distCoeffs from left.yaml / right.yaml.
    bool loadIntrinsics(cv::Mat& K_left,  cv::Mat& dist_left,
                        cv::Mat& K_right, cv::Mat& dist_right,
                        cv::Size& imageSize) const;

    // Detect corners simultaneously in one stereo pair.
    // Returns true only if BOTH images produce a full set of corners.
    bool detectPair(const std::string& leftPath,
                    const std::string& rightPath,
                    std::vector<cv::Point2f>& cornersL,
                    std::vector<cv::Point2f>& cornersR,
                    cv::Size& imageSize) const;

    std::vector<cv::Point3f> buildObjectPoints() const;

    void validateResult(const Result& result) const;
};