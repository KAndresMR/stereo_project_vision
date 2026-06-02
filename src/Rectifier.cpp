#include "Rectifier.hpp"
#include "DiagnosticLogger.hpp"
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
Rectifier::Rectifier(const CalibrationConfig& config)
    : config_(config) {}

// ─────────────────────────────────────────────────────────────────────────────
// loadYAMLs
//
// Reads ALL calibration data from the three YAML files produced by the
// previous two phases:
//   left.yaml   → K_left, dist_left
//   right.yaml  → K_right, dist_right
//   stereo.yaml → R, T, R1, R2, P1, P2, Q (pre-computed by StereoCalibrator)
//
// Note: R1, R2, P1, P2, Q are already stored in stereo.yaml because
// StereoCalibrator runs stereoRectify before saving. This avoids re-running
// stereoRectify here with potentially different parameters.
// ─────────────────────────────────────────────────────────────────────────────
bool Rectifier::loadYAMLs(cv::Mat& K_left,  cv::Mat& dist_left,
                           cv::Mat& K_right, cv::Mat& dist_right,
                           cv::Mat& R,       cv::Mat& T,
                           cv::Mat& R1,      cv::Mat& R2,
                           cv::Mat& P1,      cv::Mat& P2,
                           cv::Mat& Q,
                           cv::Size& imageSize) const {
    // ── left.yaml ─────────────────────────────────────────────────────────────
    {
        cv::FileStorage fs(config_.leftYaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("Rectifier", "Cannot open: " + config_.leftYaml);
            return false;
        }
        fs["camera_matrix"]           >> K_left;
        fs["distortion_coefficients"] >> dist_left;
        Log::yamlLoaded(config_.leftYaml, true);
    }

    // ── right.yaml ────────────────────────────────────────────────────────────
    {
        cv::FileStorage fs(config_.rightYaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("Rectifier", "Cannot open: " + config_.rightYaml);
            return false;
        }
        fs["camera_matrix"]           >> K_right;
        fs["distortion_coefficients"] >> dist_right;
        Log::yamlLoaded(config_.rightYaml, true);
    }

    // ── stereo.yaml ───────────────────────────────────────────────────────────
    {
        cv::FileStorage fs(config_.stereoYaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("Rectifier", "Cannot open: " + config_.stereoYaml);
            Log::error("Rectifier", "Run stereo calibration first (menu option 4).");
            return false;
        }
        fs["R"]  >> R;   fs["T"]  >> T;
        fs["R1"] >> R1;  fs["R2"] >> R2;
        fs["P1"] >> P1;  fs["P2"] >> P2;
        fs["Q"]  >> Q;
        int w = 0, h = 0;
        fs["image_width"] >> w; fs["image_height"] >> h;
        if (w > 0 && h > 0) imageSize = {w, h};
        Log::yamlLoaded(config_.stereoYaml, true);
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute
//
// Builds the pixel remap tables (map1x, map1y, map2x, map2y).
//
// How remapping works:
//   For every output pixel (u, v) in the rectified image, the remap table
//   stores the source pixel (u', v') in the raw distorted image.
//   cv::remap() then samples the raw image at (u', v') using bilinear
//   interpolation to fill (u, v) in the output.
//
//   This is a one-time computation: the tables are stored and reused for
//   every frame at runtime (very fast — just a lookup per pixel).
// ─────────────────────────────────────────────────────────────────────────────
bool Rectifier::compute() {
    Log::separator("Rectifier — compute");

    cv::Mat K_left, dist_left, K_right, dist_right, R, T;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imageSize;

    if (!loadYAMLs(K_left, dist_left, K_right, dist_right,
                   R, T, R1, R2, P1, P2, Q, imageSize)) {
        return false;
    }

    maps_.R1 = R1;  maps_.R2 = R2;
    maps_.P1 = P1;  maps_.P2 = P2;
    maps_.Q  = Q;
    maps_.imageSize = imageSize;

    // ── initUndistortRectifyMap ───────────────────────────────────────────────
    //
    // Combines two operations into one remap table:
    //   1. Undistortion: removes lens distortion using K and dist
    //   2. Rectification: applies R1 (or R2) to align epipolar lines
    //
    // CV_32FC1: floating-point map — sub-pixel accuracy in the lookup.
    //
    cv::initUndistortRectifyMap(
        K_left, dist_left, R1, P1,
        imageSize, CV_32FC1,
        maps_.map1x, maps_.map1y
    );

    cv::initUndistortRectifyMap(
        K_right, dist_right, R2, P2,
        imageSize, CV_32FC1,
        maps_.map2x, maps_.map2y
    );

    maps_.ready = true;

    Log::info("Rectifier", "Remap tables computed for " +
              std::to_string(imageSize.width) + "×" +
              std::to_string(imageSize.height));
    Log::info("Rectifier",
        "Q[3][2] = " + std::to_string(-1.0 / Q.at<double>(3,2)) +
        " m (baseline encoded)");
    Log::info("Rectifier",
        "Q[2][3] = " + std::to_string(Q.at<double>(2,3)) +
        " px (focal length encoded)");

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// rectify
//
// Applies the pre-computed remap tables to a raw stereo pair.
// Input frames can be either from disk or from the live stream.
//
// cv::INTER_LINEAR: bilinear interpolation — good balance of quality/speed.
// cv::INTER_LANCZOS4: higher quality but slower (use for debug/offline).
// ─────────────────────────────────────────────────────────────────────────────
std::pair<cv::Mat, cv::Mat> Rectifier::rectify(const cv::Mat& rawLeft,
                                                const cv::Mat& rawRight) const {
    if (!maps_.ready) {
        Log::error("Rectifier", "Maps not computed. Call compute() first.");
        return {};
    }

    cv::Mat rectLeft, rectRight;
    cv::remap(rawLeft,  rectLeft,  maps_.map1x, maps_.map1y, cv::INTER_LINEAR);
    cv::remap(rawRight, rectRight, maps_.map2x, maps_.map2y, cv::INTER_LINEAR);

    return {rectLeft, rectRight};
}

// ─────────────────────────────────────────────────────────────────────────────
// drawEpipolarLines
//
// Creates a side-by-side composite image with horizontal green lines.
//
// How to interpret:
//   GOOD: a distinctive feature (corner, edge) in the left image at row Y
//         appears at the SAME row Y in the right image.
//   BAD:  the same feature appears at different rows → rectification failed
//         or calibration was poor.
//
// Typical causes of bad alignment:
//   - stereo.yaml computed with a different image size than current frames
//   - cameras were physically moved after calibration
//   - poor stereo calibration (high RMS or too few pairs)
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat Rectifier::drawEpipolarLines(const cv::Mat& rectLeft,
                                      const cv::Mat& rectRight,
                                      int lineSpacing) const {
    // Resize to 640×480 if needed (for consistent display)
    cv::Mat dispL = rectLeft.clone();
    cv::Mat dispR = rectRight.clone();

    // Side-by-side
    cv::Mat combined;
    cv::hconcat(dispL, dispR, combined);

    // Draw horizontal epipolar lines
    const cv::Scalar lineColor(0, 220, 0);  // green
    for (int y = lineSpacing; y < combined.rows; y += lineSpacing) {
        cv::line(combined, {0, y}, {combined.cols, y}, lineColor, 1);
        // Row number label (every other line to avoid clutter)
        if ((y / lineSpacing) % 2 == 0) {
            cv::putText(combined, std::to_string(y),
                        {2, y - 3}, cv::FONT_HERSHEY_SIMPLEX,
                        0.35, lineColor, 1);
        }
    }

    // Labels
    cv::putText(combined, "LEFT (rectified)",
                {10, 18}, cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 220, 255), 1);
    cv::putText(combined, "RIGHT (rectified)",
                {rectLeft.cols + 10, 18}, cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 220, 255), 1);
    cv::putText(combined, "Lines should pass same features in both images",
                {10, combined.rows - 8}, cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(180, 180, 180), 1);

    return combined;
}

// ─────────────────────────────────────────────────────────────────────────────
// previewDataset
//
// Offline validation: load each saved stereo pair, rectify, and display
// with epipolar lines. Good for checking calibration quality before
// connecting live cameras.
// ─────────────────────────────────────────────────────────────────────────────
void Rectifier::previewDataset() const {
    if (!maps_.ready) {
        Log::error("Rectifier", "Call compute() before previewDataset().");
        return;
    }

    std::vector<std::string> leftPaths, rightPaths;
    cv::glob(config_.datasetDir + "/left/*.jpg",  leftPaths,  false);
    cv::glob(config_.datasetDir + "/right/*.jpg", rightPaths, false);
    std::sort(leftPaths.begin(),  leftPaths.end());
    std::sort(rightPaths.begin(), rightPaths.end());

    int n = static_cast<int>(std::min(leftPaths.size(), rightPaths.size()));
    if (n == 0) {
        Log::error("Rectifier", "No image pairs found in dataset.");
        return;
    }

    Log::info("Rectifier",
        "Offline preview: " + std::to_string(n) +
        " pairs. ANY KEY = next | ESC = exit");

    for (int i = 0; i < n; ++i) {
        cv::Mat rawL = cv::imread(leftPaths[i],  cv::IMREAD_COLOR);
        cv::Mat rawR = cv::imread(rightPaths[i], cv::IMREAD_COLOR);
        if (rawL.empty() || rawR.empty()) continue;

        auto [rectL, rectR] = rectify(rawL, rawR);
        cv::Mat debug = drawEpipolarLines(rectL, rectR);

        // Add pair index label
        std::string label = "Pair " + std::to_string(i+1) + "/" +
                            std::to_string(n) + "  — " +
                            std::filesystem::path(leftPaths[i]).filename().string();
        cv::putText(debug, label,
                    {debug.cols/2 - 180, 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1);

        // Scale for display if too wide
        double scale = std::min(1.0, 1400.0 / debug.cols);
        if (scale < 1.0) cv::resize(debug, debug, {}, scale, scale);

        cv::imshow("Epipolar Check", debug);
        int key = cv::waitKey(0);
        if (key == 27) break;  // ESC
    }

    cv::destroyWindow("Epipolar Check");
}