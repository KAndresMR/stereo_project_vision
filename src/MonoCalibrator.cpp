#include "MonoCalibrator.hpp"
#include "DiagnosticLogger.hpp"
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
MonoCalibrator::MonoCalibrator(const CalibrationConfig& config)
    : config_(config)
    , detector_(config) {}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

std::string MonoCalibrator::sideName(Side side) const {
    return (side == Side::LEFT) ? "LEFT" : "RIGHT";
}

std::string MonoCalibrator::imageDir(Side side) const {
    return config_.datasetDir + "/" + ((side == Side::LEFT) ? "left" : "right");
}

std::string MonoCalibrator::yamlPath(Side side) const {
    return (side == Side::LEFT) ? config_.leftYaml : config_.rightYaml;
}

// ─────────────────────────────────────────────────────────────────────────────
// buildObjectPoints
//
// Creates the "ground truth" 3-D positions of the chessboard corners in the
// board's own coordinate system. Since the board is flat, Z = 0 for every point.
//
// For boardSize = {8, 5} and squareSizeM = 0.025f, the points are:
//   (0.000, 0.000, 0)  (0.025, 0.000, 0)  ...  (0.175, 0.000, 0)
//   (0.000, 0.025, 0)  (0.025, 0.025, 0)  ...  (0.175, 0.025, 0)
//   ...
//   (0.000, 0.100, 0)  (0.025, 0.100, 0)  ...  (0.175, 0.100, 0)
//
// These are in METERS because squareSizeM is in meters. This makes the
// translation vector T from stereoCalibrate come out in meters too — so
// later, Z = f*B/d gives depth in meters directly.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<cv::Point3f> MonoCalibrator::buildObjectPoints() const {
    std::vector<cv::Point3f> pts;
    pts.reserve(config_.boardSize.width * config_.boardSize.height);

    for (int row = 0; row < config_.boardSize.height; ++row) {
        for (int col = 0; col < config_.boardSize.width; ++col) {
            pts.push_back({
                col * config_.squareSizeM,
                row * config_.squareSizeM,
                0.0f
            });
        }
    }
    return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// loadAndDetect
//
// Loads one JPEG from disk, converts to grayscale, runs ChessboardDetector
// (which includes CLAHE if needed and sub-pixel refinement).
//
// Returns true only if ALL expected corners were found.
// The image size is recorded from the first valid image — all subsequent
// images must match, which calibrateCamera requires.
// ─────────────────────────────────────────────────────────────────────────────
bool MonoCalibrator::loadAndDetect(const std::string&        imagePath,
                                   std::vector<cv::Point2f>& outCorners,
                                   cv::Size&                 outImageSize) const {
    // ── Load ─────────────────────────────────────────────────────────────────
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        Log::error("MonoCalib", "Cannot read: " + imagePath);
        return false;
    }

    outImageSize = img.size();

    // ── Convert to gray ───────────────────────────────────────────────────────
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // ── Detect (CLAHE + findChessboardCorners + cornerSubPix) ────────────────
    cv::Mat displayDummy = img;   // detector draws on this; we discard it here
    DetectionResult det  = detector_.detect(gray, displayDummy);

    // Log every image — but only one line per image (not spam)
    const std::string fname = fs::path(imagePath).filename().string();
    if (det.found) {
        Log::info("MonoCalib",
            "  ✓ " + fname +
            " | brightness=" + std::to_string(static_cast<int>(det.brightness)) +
            (det.claheUsed ? " | CLAHE=ON" : ""));
    } else {
        Log::warn("MonoCalib",
            "  ✗ " + fname +
            " | brightness=" + std::to_string(static_cast<int>(det.brightness)) +
            " | corners=" + std::to_string(det.cornersFound) +
            "/" + std::to_string(det.cornersExpected) +
            (det.claheUsed ? " | CLAHE=ON" : "") +
            " → SKIPPED");
    }

    if (!det.found) return false;

    outCorners = det.corners;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// validateResult
//
// Checks that the calibration output values are physically plausible for an
// OV2640 sensor at VGA (640×480). Warns but does NOT fail — the caller decides
// whether to accept the result.
//
// Expected ranges for OV2640 VGA:
//   fx, fy : 350–750 px   (focal length in pixels; related to FOV)
//   cx     : 270–370 px   (principal point ≈ horizontal center)
//   cy     : 190–290 px   (principal point ≈ vertical center)
//   k1     : -0.8 to 0.1  (barrel distortion is negative; OV2640 typically -0.3 to -0.6)
//   RMS    : < 1.0 acceptable, < 0.5 good, < 0.3 excellent
// ─────────────────────────────────────────────────────────────────────────────
void MonoCalibrator::validateResult(const Result& result) const {

    const cv::Mat& K  = result.cameraMatrix;
    const cv::Mat& D  = result.distCoeffs;

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double k1 = D.at<double>(0);

    const std::string tag = "Validate";

    // ── fx / fy ───────────────────────────────────────────────────────────────
    if (fx < 300 || fx > 900)
        Log::warn(tag, "fx=" + std::to_string(fx) + " is outside [300,900] — suspicious");
    if (fy < 300 || fy > 900)
        Log::warn(tag, "fy=" + std::to_string(fy) + " is outside [300,900] — suspicious");

    // fx and fy should be close (< 5% difference for square pixels)
    if (std::abs(fx - fy) / std::max(fx, fy) > 0.05)
        Log::warn(tag, "fx and fy differ by >5% — possible calibration issue");

    // ── Principal point ───────────────────────────────────────────────────────
    if (cx < 200 || cx > 440)
        Log::warn(tag, "cx=" + std::to_string(cx) + " is far from image center (320)");
    if (cy < 150 || cy > 330)
        Log::warn(tag, "cy=" + std::to_string(cy) + " is far from image center (240)");

    // ── k1 (main distortion coefficient) ─────────────────────────────────────
    if (k1 > 0.1)
        Log::warn(tag, "k1=" + std::to_string(k1) +
                  " is positive (pincushion). OV2640 usually has barrel (negative).");
    if (k1 < -0.9)
        Log::warn(tag, "k1=" + std::to_string(k1) +
                  " is very negative. Likely a bad dataset — consider recapturing.");

    // ── RMS ───────────────────────────────────────────────────────────────────
    if (result.rpe > 1.0)
        Log::warn(tag, "RMS=" + std::to_string(result.rpe) +
                  " > 1.0 px. Recapture recommended (more angles, less blur).");

    // ── Image count ───────────────────────────────────────────────────────────
    if (result.imagesUsed < 10)
        Log::warn(tag, "Only " + std::to_string(result.imagesUsed) +
                  " images used — need at least 10 for stable calibration.");

    if (result.rpe < 0.5 && result.imagesUsed >= 15)
        Log::info(tag, "Result looks good ✓");
}

// ─────────────────────────────────────────────────────────────────────────────
// calibrate — the main pipeline
// ─────────────────────────────────────────────────────────────────────────────
MonoCalibrator::Result MonoCalibrator::calibrate(Side side) const {

    Result result;
    const std::string name = sideName(side);
    const std::string dir  = imageDir(side);

    Log::separator("MonoCalibrator — " + name + " camera");

    // ── Step 1: find image files ──────────────────────────────────────────────
    if (!fs::exists(dir)) {
        Log::error("MonoCalib", "Directory not found: " + dir);
        return result;
    }

    std::vector<std::string> imagePaths;
    cv::glob(dir + "/*.jpg", imagePaths, false);
    std::sort(imagePaths.begin(), imagePaths.end());  // consistent order

    result.imagesTotal = static_cast<int>(imagePaths.size());
    Log::info("MonoCalib", "Found " + std::to_string(result.imagesTotal) +
              " images in " + dir);

    if (result.imagesTotal == 0) {
        Log::error("MonoCalib", "No .jpg images found. Did the capture phase run?");
        return result;
    }

    // ── Step 2: build known 3-D object points ────────────────────────────────
    // Same for every image — the board geometry does not change.
    const std::vector<cv::Point3f> singleObjPts = buildObjectPoints();
    Log::info("MonoCalib",
              "Object points per image: " + std::to_string(singleObjPts.size()) +
              " (board " +
              std::to_string(config_.boardSize.width) + "×" +
              std::to_string(config_.boardSize.height) + " = " +
              std::to_string(config_.boardSize.width * config_.boardSize.height) +
              " corners)");

    // ── Step 3: detect corners in every image ────────────────────────────────
    Log::info("MonoCalib", "Detecting corners...");

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;

    for (const auto& path : imagePaths) {
        std::vector<cv::Point2f> corners;
        cv::Size sz;

        if (loadAndDetect(path, corners, sz)) {
            // Record image size from first valid image.
            // calibrateCamera requires all images to have the same size.
            if (imageSize.empty()) imageSize = sz;

            if (sz != imageSize) {
                Log::warn("MonoCalib",
                    fs::path(path).filename().string() +
                    " has different size — skipping");
                result.skippedImages.push_back(path);
                continue;
            }

            objectPoints.push_back(singleObjPts);
            imagePoints.push_back(corners);
        } else {
            result.skippedImages.push_back(path);
        }
    }

    result.imagesUsed = static_cast<int>(imagePoints.size());
    Log::info("MonoCalib",
              "Valid images: " + std::to_string(result.imagesUsed) +
              "/" + std::to_string(result.imagesTotal) +
              "  |  Skipped: " + std::to_string(result.skippedImages.size()));

    if (result.imagesUsed < 6) {
        Log::error("MonoCalib",
            "Need at least 6 valid images, got " +
            std::to_string(result.imagesUsed) + ". Aborting.");
        return result;
    }

    // ── Step 4: run calibrateCamera ───────────────────────────────────────────
    //
    // calibrateCamera solves for K and distCoeffs by minimizing the total
    // reprojection error across all corners in all images.
    //
    // Flags used:
    //   (none / 0) — estimate all 5 standard distortion coefficients
    //                (k1, k2, p1, p2, k3). For OV2640 this is usually enough.
    //
    // Output rotationVecs and translationVecs describe the pose of the board
    // in each image. We don't save them — they're only needed internally.
    //
    Log::info("MonoCalib", "Running cv::calibrateCamera...");

    std::vector<cv::Mat> rvecs, tvecs;

    try {
        result.rpe = cv::calibrateCamera(
            objectPoints,
            imagePoints,
            imageSize,
            result.cameraMatrix,
            result.distCoeffs,
            rvecs,
            tvecs
            // No extra flags: use the default 5-coefficient model.
            // To enable 8-coefficient rational model (better for wide-angle),
            // add: cv::CALIB_RATIONAL_MODEL
        );
    } catch (const cv::Exception& e) {
        Log::error("MonoCalib", "calibrateCamera threw: " + std::string(e.what()));
        return result;
    }

    result.imageSize = imageSize;
    result.success   = true;

    // ── Step 5: validate and log ──────────────────────────────────────────────
    validateResult(result);
    printSummary(result, side);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// saveYAML
// ─────────────────────────────────────────────────────────────────────────────
bool MonoCalibrator::saveYAML(const Result& result, Side side) const {

    if (!result.success) {
        Log::error("MonoCalib", "Cannot save YAML — calibration was not successful.");
        return false;
    }

    const std::string path = yamlPath(side);

    // Create parent directory if needed
    fs::create_directories(fs::path(path).parent_path());

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        Log::error("MonoCalib", "Cannot open for writing: " + path);
        return false;
    }

    // ── Write metadata ────────────────────────────────────────────────────────
    fs << "camera_name"            << sideName(side)
       << "image_width"            << result.imageSize.width
       << "image_height"           << result.imageSize.height
       << "board_width"            << config_.boardSize.width
       << "board_height"           << config_.boardSize.height
       << "square_size_m"          << config_.squareSizeM
       << "images_used"            << result.imagesUsed
       << "images_total"           << result.imagesTotal
       << "rms_reprojection_error" << result.rpe;

    // ── Write calibration matrices ────────────────────────────────────────────
    fs << "camera_matrix"              << result.cameraMatrix
       << "distortion_coefficients"    << result.distCoeffs;

    fs.release();

    Log::yamlSaved(path);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// printSummary
// ─────────────────────────────────────────────────────────────────────────────
void MonoCalibrator::printSummary(const Result& result, Side side) const {
    if (!result.success) return;

    const cv::Mat& K = result.cameraMatrix;
    const cv::Mat& D = result.distCoeffs;
    double fx = K.at<double>(0,0), fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2), cy = K.at<double>(1,2);

    Log::separator("Result — " + sideName(side));
    std::cout << std::fixed << std::setprecision(2)
        << "\n  Camera matrix K:\n"
        << "    | " << std::setw(8) << fx << "   0.00  " << std::setw(8) << cx << " |\n"
        << "    |    0.00  "          << std::setw(8) << fy << "  " << std::setw(8) << cy << " |\n"
        << "    |    0.00     0.00     1.00 |\n\n"
        << "  Distortion [k1 k2 p1 p2 k3]:  [ ";
    for (int i = 0; i < D.cols; ++i)
        std::cout << std::setw(9) << std::setprecision(5) << D.at<double>(i) << " ";
    std::cout << "]\n\n"
        << "  RMS  : " << std::setprecision(4) << result.rpe << " px\n"
        << "  Used : " << result.imagesUsed << " / " << result.imagesTotal << "\n"
        << "  Size : " << result.imageSize.width << "x" << result.imageSize.height << "\n\n";
}
