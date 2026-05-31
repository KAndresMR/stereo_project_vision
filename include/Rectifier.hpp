#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "CalibrationConfig.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Rectifier — Phase 3 of the pipeline
//
// Responsibility:
//   Given the stereo calibration (R, T, K, dist for each camera),
//   compute the pixel-level remapping that transforms raw camera images
//   into rectified images where:
//     - epipolar lines are perfectly horizontal
//     - corresponding points in left and right images lie on the same row
//     - stereo matching (SGBM) can scan only one row per pixel
//
// Two usage modes:
//   OFFLINE: rectify saved images from the dataset (for debug/verification)
//   LIVE:    rectify frames coming from the ESP32 streams in real time
//
// Usage:
//   Rectifier rect(config);
//   if (rect.compute()) {
//       auto [left_r, right_r] = rect.rectify(rawLeft, rawRight);
//       cv::Mat debug = rect.drawEpipolarLines(left_r, right_r);
//   }
// ─────────────────────────────────────────────────────────────────────────────
class Rectifier {
public:

    // All pre-computed rectification data needed at runtime.
    struct Maps {
        cv::Mat map1x, map1y;   // pixel remap for LEFT camera
        cv::Mat map2x, map2y;   // pixel remap for RIGHT camera
        cv::Mat R1, R2;         // rectifying rotation for each camera
        cv::Mat P1, P2;         // projection matrices after rectification
        cv::Mat Q;              // 4×4 disparity-to-depth matrix
        cv::Size imageSize;
        bool ready = false;
    };

    explicit Rectifier(const CalibrationConfig& config);

    // Load all YAMLs and compute remap tables.
    // Must be called before rectify() or drawEpipolarLines().
    bool compute();

    // Apply rectification to a stereo pair.
    // Input:  raw frames from cameras (or from disk)
    // Output: pair of rectified images (left, right)
    std::pair<cv::Mat, cv::Mat> rectify(const cv::Mat& rawLeft,
                                         const cv::Mat& rawRight) const;

    // Debug visualization: side-by-side rectified pair with horizontal
    // green lines overlaid. If rectification is correct, the same physical
    // point appears on the SAME line in both images.
    cv::Mat drawEpipolarLines(const cv::Mat& rectLeft,
                               const cv::Mat& rectRight,
                               int lineSpacing = 40) const;

    // Run an offline visual check on the saved dataset pairs.
    // Loads each pair, rectifies, draws epipolar lines, shows window.
    // Press any key to advance, ESC to quit.
    void previewDataset() const;

    bool isReady() const { return maps_.ready; }
    const Maps& maps() const { return maps_; }

private:
    CalibrationConfig config_;
    Maps maps_;

    bool loadYAMLs(cv::Mat& K_left,  cv::Mat& dist_left,
                   cv::Mat& K_right, cv::Mat& dist_right,
                   cv::Mat& R,       cv::Mat& T,
                   cv::Mat& R1,      cv::Mat& R2,
                   cv::Mat& P1,      cv::Mat& P2,
                   cv::Mat& Q,
                   cv::Size& imageSize) const;
};