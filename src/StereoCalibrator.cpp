#include "StereoCalibrator.hpp"
#include "DiagnosticLogger.hpp"
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
StereoCalibrator::StereoCalibrator(const CalibrationConfig& config)
    : config_(config)
    , detector_(config) {}

// ─────────────────────────────────────────────────────────────────────────────
// buildObjectPoints
// ─────────────────────────────────────────────────────────────────────────────
std::vector<cv::Point3f> StereoCalibrator::buildObjectPoints() const {
    std::vector<cv::Point3f> pts;
    pts.reserve(config_.boardSize.width * config_.boardSize.height);
    for (int r = 0; r < config_.boardSize.height; ++r)
        for (int c = 0; c < config_.boardSize.width; ++c)
            pts.push_back({c * config_.squareSizeM,
                           r * config_.squareSizeM,
                           0.0f});
    return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// loadIntrinsics
//
// Reads K and distCoeffs for each camera from the individual YAML files
// produced by MonoCalibrator.
// ─────────────────────────────────────────────────────────────────────────────
bool StereoCalibrator::loadIntrinsics(cv::Mat& K_left,  cv::Mat& dist_left,
                                      cv::Mat& K_right, cv::Mat& dist_right,
                                      cv::Size& imageSize) const {
    auto load = [&](const std::string& path,
                    cv::Mat& K, cv::Mat& dist,
                    const std::string& name) -> bool {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("StereoCalib", "Cannot open: " + path);
            Log::error("StereoCalib", "Run mono calibration first (menu option 3).");
            return false;
        }
        fs["camera_matrix"]           >> K;
        fs["distortion_coefficients"] >> dist;
        int w = 0, h = 0;
        fs["image_width"]  >> w;
        fs["image_height"] >> h;
        if (w > 0 && h > 0) imageSize = {w, h};
        fs.release();
        Log::yamlLoaded(path, true);
        Log::info("StereoCalib",
            name + ": fx=" + std::to_string((int)K.at<double>(0,0)) +
            " fy=" + std::to_string((int)K.at<double>(1,1)) +
            " cx=" + std::to_string((int)K.at<double>(0,2)) +
            " cy=" + std::to_string((int)K.at<double>(1,2)));
        return true;
    };

    return load(config_.leftYaml,  K_left,  dist_left,  "LEFT ")
        && load(config_.rightYaml, K_right, dist_right, "RIGHT");
}

// ─────────────────────────────────────────────────────────────────────────────
// detectPair
//
// Loads one stereo pair from disk, detects corners in BOTH images.
// Only returns true when BOTH cameras find the full pattern.
// A pair where only one camera succeeds is useless for stereoCalibrate
// because we need corresponding 2D points in left AND right.
// ─────────────────────────────────────────────────────────────────────────────
bool StereoCalibrator::detectPair(const std::string& leftPath,
                                  const std::string& rightPath,
                                  std::vector<cv::Point2f>& cornersL,
                                  std::vector<cv::Point2f>& cornersR,
                                  cv::Size& imageSize) const {
    cv::Mat imgL = cv::imread(leftPath,  cv::IMREAD_COLOR);
    cv::Mat imgR = cv::imread(rightPath, cv::IMREAD_COLOR);

    if (imgL.empty() || imgR.empty()) {
        Log::error("StereoCalib", "Cannot read pair: " +
                   fs::path(leftPath).filename().string());
        return false;
    }

    if (imgL.size() != imgR.size()) {
        Log::warn("StereoCalib",
            "Image size mismatch between LEFT and RIGHT.");
        return false;
    }

    imageSize = imgL.size();

    cv::Mat grayL, grayR;
    cv::cvtColor(imgL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, grayR, cv::COLOR_BGR2GRAY);

    cv::Mat dummyL = imgL, dummyR = imgR;
    DetectionResult dL = detector_.detect(grayL, dummyL);
    DetectionResult dR = detector_.detect(grayR, dummyR);

    std::string fname = fs::path(leftPath).filename().string();

    if (dL.found && dR.found) {
        cornersL = dL.corners;
        cornersR = dR.corners;
        Log::info("StereoCalib",
            "  ✓ " + fname +
            " | L_brightness=" + std::to_string((int)dL.brightness) +
            " R_brightness=" + std::to_string((int)dR.brightness));
        return true;
    }

    // Explain which side failed
    std::string reason;
    if (!dL.found && !dR.found) reason = "BOTH cameras failed";
    else if (!dL.found)         reason = "LEFT failed (corners=" +
                                         std::to_string(dL.cornersFound) + "/"+
                                         std::to_string(dL.cornersExpected)+")";
    else                         reason = "RIGHT failed (corners=" +
                                         std::to_string(dR.cornersFound) + "/"+
                                         std::to_string(dR.cornersExpected)+")";

    Log::warn("StereoCalib", "  ✗ " + fname + " | " + reason + " → SKIPPED");
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// validateResult
// ─────────────────────────────────────────────────────────────────────────────
void StereoCalibrator::validateResult(const Result& result) const {
    const std::string tag = "Validate";

    // ── Baseline ──────────────────────────────────────────────────────────────
    // The baseline should be close to the physical separation of the cameras.
    // For your ~70mm setup: 0.05–0.10 m is the expected range.
    double blMm = result.baselineM * 1000.0;
    Log::info(tag, "Baseline: " + std::to_string(blMm) + " mm");
    if (result.baselineM < 0.03)
        Log::warn(tag, "Baseline < 30mm — cameras may be too close, or T is wrong");
    else if (result.baselineM > 0.20)
        Log::warn(tag, "Baseline > 200mm — unusually large. Check camera positions.");
    else
        Log::info(tag, "Baseline in expected range ✓");

    // ── RMS ───────────────────────────────────────────────────────────────────
    if      (result.rpe < 0.3)  Log::info(tag, "RMS=" + std::to_string(result.rpe) + " ✓✓ Excellent");
    else if (result.rpe < 0.5)  Log::info(tag, "RMS=" + std::to_string(result.rpe) + " ✓  Good");
    else if (result.rpe < 1.0)  Log::warn(tag, "RMS=" + std::to_string(result.rpe) + " △  Acceptable");
    else                        Log::warn(tag, "RMS=" + std::to_string(result.rpe) + " ✗  Poor — recapture recommended");

    // ── T direction ──────────────────────────────────────────────────────────
    // T[0] should dominate (cameras separated horizontally).
    // T[1] and T[2] should be small fractions of T[0].
    double tx = std::abs(result.T.at<double>(0));
    double ty = std::abs(result.T.at<double>(1));
    double tz = std::abs(result.T.at<double>(2));
    if (ty > tx * 0.3 || tz > tx * 0.3)
        Log::warn(tag, "T has significant vertical/depth component — cameras may not be co-planar");
    else
        Log::info(tag, "T direction OK (primarily horizontal) ✓");

    // ── R (should be close to identity) ──────────────────────────────────────
    // Extract rotation angles using Rodrigues
    cv::Mat rvec;
    cv::Rodrigues(result.R, rvec);
    double angleRad = cv::norm(rvec);
    double angleDeg = angleRad * 180.0 / CV_PI;
    Log::info(tag, "Rotation angle: " + std::to_string(angleDeg) + " deg");
    if (angleDeg > 5.0)
        Log::warn(tag, "Rotation > 5° between cameras — physically large misalignment");

    // ── Pairs ─────────────────────────────────────────────────────────────────
    if (result.pairsUsed < 10)
        Log::warn(tag, "Only " + std::to_string(result.pairsUsed) +
                  " pairs used — aim for 15+");
}

// ─────────────────────────────────────────────────────────────────────────────
// calibrate — the main pipeline
// ─────────────────────────────────────────────────────────────────────────────
StereoCalibrator::Result StereoCalibrator::calibrate() const {
    Result result;

    Log::separator("StereoCalibrator — Phase 2");

    // ── Step 1: load individual intrinsics ────────────────────────────────────
    cv::Mat K_left, dist_left, K_right, dist_right;
    cv::Size imageSize;

    if (!loadIntrinsics(K_left, dist_left, K_right, dist_right, imageSize)) {
        return result;
    }

    // ── Step 2: find matching image pairs ────────────────────────────────────
    std::vector<std::string> leftPaths, rightPaths;
    cv::glob(config_.datasetDir + "/left/*.jpg",  leftPaths,  false);
    cv::glob(config_.datasetDir + "/right/*.jpg", rightPaths, false);
    std::sort(leftPaths.begin(),  leftPaths.end());
    std::sort(rightPaths.begin(), rightPaths.end());

    result.pairsTotal = static_cast<int>(
        std::min(leftPaths.size(), rightPaths.size()));

    Log::info("StereoCalib",
        "Image pairs found: " + std::to_string(result.pairsTotal));

    if (result.pairsTotal == 0) {
        Log::error("StereoCalib", "No image pairs found. Run capture first.");
        return result;
    }

    // ── Step 3: detect corners in each synchronized pair ─────────────────────
    Log::info("StereoCalib", "Detecting corners in stereo pairs...");

    const auto singleObjPts  = buildObjectPoints();
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imgPtsL, imgPtsR;

    for (int i = 0; i < result.pairsTotal; ++i) {
        std::vector<cv::Point2f> cL, cR;
        cv::Size sz;

        if (detectPair(leftPaths[i], rightPaths[i], cL, cR, sz)) {
            if (imageSize.empty()) imageSize = sz;
            objectPoints.push_back(singleObjPts);
            imgPtsL.push_back(cL);
            imgPtsR.push_back(cR);
        }
    }

    result.pairsUsed = static_cast<int>(objectPoints.size());
    Log::info("StereoCalib",
        "Valid stereo pairs: " + std::to_string(result.pairsUsed) +
        " / " + std::to_string(result.pairsTotal));

    if (result.pairsUsed < 6) {
        Log::error("StereoCalib",
            "Need at least 6 valid stereo pairs, got " +
            std::to_string(result.pairsUsed));
        return result;
    }

    // ── Step 4: stereoCalibrate ───────────────────────────────────────────────
    //
    // CALIB_FIX_INTRINSIC: do NOT touch K_left / K_right / dist.
    // We already calibrated them individually. Here we only solve for R and T
    // (the 6-DoF rigid body transform from left camera to right camera).
    //
    // TermCriteria: allow up to 100 iterations or until change < 1e-6.
    //
    Log::info("StereoCalib", "Running cv::stereoCalibrate...");

    try {
        result.rpe = cv::stereoCalibrate(
            objectPoints,
            imgPtsL, imgPtsR,
            K_left,  dist_left,
            K_right, dist_right,
            imageSize,
            result.R, result.T,
            result.E, result.F,
            cv::CALIB_FIX_INTRINSIC,
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                100, 1e-6)
        );
    } catch (const cv::Exception& e) {
        Log::error("StereoCalib", "stereoCalibrate threw: " +
                   std::string(e.what()));
        return result;
    }

    // ── Step 5: compute baseline ──────────────────────────────────────────────
    result.baselineM = cv::norm(result.T);
    result.imageSize = imageSize;
    result.success   = true;

    validateResult(result);
    printSummary(result);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// saveYAML
//
// Saves R, T, E, F from stereoCalibrate, plus R1, R2, P1, P2, Q from
// stereoRectify — everything the next phase (Rectifier) needs in one file.
// ─────────────────────────────────────────────────────────────────────────────
bool StereoCalibrator::saveYAML(const Result& result) const {
    if (!result.success) {
        Log::error("StereoCalib", "Cannot save — calibration was not successful.");
        return false;
    }

    // ── Re-load intrinsics to run stereoRectify ───────────────────────────────
    cv::Mat K_left, dist_left, K_right, dist_right;
    cv::Size imageSize;
    if (!loadIntrinsics(K_left, dist_left, K_right, dist_right, imageSize)) {
        Log::error("StereoCalib", "Cannot reload intrinsics for rectify step.");
        return false;
    }
    if (imageSize.empty()) imageSize = result.imageSize;

    // ── stereoRectify ─────────────────────────────────────────────────────────
    //
    // Computes the rotation matrices (R1, R2) and projection matrices (P1, P2)
    // that make the epipolar lines horizontal and co-planar.
    // Q is the 4×4 disparity-to-depth reprojection matrix used later:
    //   Z = f * B / d   is embedded in Q.
    //
    // alpha = 0:  all pixels in rectified image are valid (some FOV is cropped)
    // alpha = 1:  all original pixels preserved (black borders visible)
    // alpha = -1: automatic (OpenCV chooses optimal crop)
    //
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(
        K_left,  dist_left,
        K_right, dist_right,
        imageSize,
        result.R, result.T,
        R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY,   // align principal points horizontally
        0,                           // alpha=0: crop to valid region
        imageSize                    // output image size = input size
    );

    // ── Write YAML ────────────────────────────────────────────────────────────
    fs::create_directories(fs::path(config_.stereoYaml).parent_path());

    cv::FileStorage fs(config_.stereoYaml, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        Log::error("StereoCalib", "Cannot write: " + config_.stereoYaml);
        return false;
    }

    fs << "image_width"            << result.imageSize.width
       << "image_height"           << result.imageSize.height
       << "pairs_used"             << result.pairsUsed
       << "pairs_total"            << result.pairsTotal
       << "rms_stereo"             << result.rpe
       << "baseline_m"             << result.baselineM;

    // Stereo geometry
    fs << "R"  << result.R
       << "T"  << result.T
       << "E"  << result.E
       << "F"  << result.F;

    // Rectification matrices (needed by Rectifier)
    fs << "R1" << R1  << "R2" << R2
       << "P1" << P1  << "P2" << P2
       << "Q"  << Q;

    fs.release();

    Log::yamlSaved(config_.stereoYaml);
    Log::info("StereoCalib",
        "Q matrix saved — baseline encoded as: " +
        std::to_string(result.baselineM * 1000.0) + " mm");

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// printSummary
// ─────────────────────────────────────────────────────────────────────────────
void StereoCalibrator::printSummary(const Result& result) const {
    if (!result.success) return;

    Log::separator("Result — Stereo Calibration");

    double tx = result.T.at<double>(0) * 1000.0;
    double ty = result.T.at<double>(1) * 1000.0;
    double tz = result.T.at<double>(2) * 1000.0;
    cv::Mat rvec;
    cv::Rodrigues(result.R, rvec);
    double angleDeg = cv::norm(rvec) * 180.0 / CV_PI;

    std::cout << std::fixed << std::setprecision(2)
        << "\n  Translation T (mm):  Tx=" << tx << "  Ty=" << ty << "  Tz=" << tz << "\n"
        << "  Baseline           : " << result.baselineM * 1000.0 << " mm\n"
        << "  Inter-camera rot   : " << std::setprecision(3) << angleDeg << " deg\n"
        << "  RMS stereo error   : " << std::setprecision(4) << result.rpe << " px\n"
        << "  Pairs used         : " << result.pairsUsed << " / " << result.pairsTotal << "\n\n";
}