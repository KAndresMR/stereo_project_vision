#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "CalibrationConfig.hpp"
#include "ChessboardDetector.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// MonoCalibrator — Phase 1 of the stereo pipeline
//
// Responsibility: given a folder of chessboard images from ONE camera,
// compute the intrinsic matrix K and distortion coefficients, then
// save them to a YAML file.
//
// What it does NOT do:
//   - capture images (that is CalibrationMode's job)
//   - stereo geometry (that is StereoCalibrator's job)
//   - rectification or disparity (later phases)
//
// Usage:
//   MonoCalibrator mc(config);
//   auto result = mc.calibrate(MonoCalibrator::Side::LEFT);
//   if (result.success) mc.saveYAML(result, MonoCalibrator::Side::LEFT);
// ─────────────────────────────────────────────────────────────────────────────
class MonoCalibrator {
public:

    // Which camera to calibrate in this run.
    enum class Side { LEFT, RIGHT };

    // ── Result struct ─────────────────────────────────────────────────────────
    struct Result {
        bool success = false;

        cv::Mat  cameraMatrix;   // 3×3 intrinsic matrix K
        cv::Mat  distCoeffs;     // distortion coefficients [k1,k2,p1,p2,k3]
        cv::Size imageSize;      // size of the images used (needed for stereoCalibrate later)

        double rpe          = 0.0;  // RMS reprojection error (lower = better)
        int    imagesUsed   = 0;    // images where corners were detected
        int    imagesTotal  = 0;    // total images found in the folder

        // Which image files were SKIPPED (failed detection) — useful for cleanup.
        std::vector<std::string> skippedImages;
    };

    // ─────────────────────────────────────────────────────────────────────────
    explicit MonoCalibrator(const CalibrationConfig& config);

    // Run the full calibration pipeline for one side.
    // Logs everything — you do not need to add any prints around this call.
    Result calibrate(Side side) const;

    // Persist result to the YAML path defined in CalibrationConfig.
    // Returns true on success.
    bool saveYAML(const Result& result, Side side) const;

    // Print a human-readable summary to stdout.
    void printSummary(const Result& result, Side side) const;

private:
    CalibrationConfig  config_;
    ChessboardDetector detector_;

    // ── Helpers ───────────────────────────────────────────────────────────────
    std::string imageDir(Side side)  const;
    std::string yamlPath(Side side)  const;
    std::string sideName(Side side)  const;  // "LEFT" or "RIGHT"

    // Build the known 3-D positions of the chessboard corners.
    // Z = 0 for all because the board is flat.
    std::vector<cv::Point3f> buildObjectPoints() const;

    // Load one image, convert to gray, detect & refine corners.
    // Returns true if a complete set of corners was found.
    bool loadAndDetect(const std::string&         imagePath,
                       std::vector<cv::Point2f>&  outCorners,
                       cv::Size&                  outImageSize) const;

    // Sanity-check the calibration result and warn if values look wrong
    // for an OV2640 sensor at VGA resolution.
    void validateResult(const Result& result) const;
};
