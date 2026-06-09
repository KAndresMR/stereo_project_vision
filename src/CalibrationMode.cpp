#include "CalibrationMode.hpp"
#include "ChessboardDetector.hpp"
#include "CalibrationSession.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

struct StereoFrames {
    cv::Mat rawLeft,  rawRight;
    cv::Mat grayLeft, grayRight;
    cv::Mat dispLeft, dispRight;
    bool valid() const { return !rawLeft.empty() && !rawRight.empty(); }
};

struct StereoDetection {
    DetectionResult left, right;
    bool bothValid = false;
};

struct PoseMetrics {
    cv::Point2f center;
    float boardWidthPx  = 0.0f;
    float boardHeightPx = 0.0f;
    float rotationDeg   = 0.0f;
    bool  valid         = false;
};

struct CoverageTracker {
    // Spatial
    bool top = false, bottom = false, left = false, right = false, center = false;
    // Scale
    bool near = false, mid = false, far = false;
    // Rotation
    bool tiltLeft = false, tiltRight = false, flat = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// acquireFrames
// ─────────────────────────────────────────────────────────────────────────────
static StereoFrames acquireFrames(CameraStream& cam1, CameraStream& cam2) {
    StereoFrames f;
    f.rawLeft  = grabFrame(cam1);
    f.rawRight = grabFrame(cam2);
    if (!f.valid()) return f;

    cv::cvtColor(f.rawLeft,  f.grayLeft,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(f.rawRight, f.grayRight, cv::COLOR_BGR2GRAY);
    f.dispLeft  = f.rawLeft.clone();
    f.dispRight = f.rawRight.clone();
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// detectStereoBoards
// ─────────────────────────────────────────────────────────────────────────────
static StereoDetection detectStereoBoards(ChessboardDetector& det, StereoFrames& f) {
    StereoDetection d;
    d.left       = det.detect(f.grayLeft,  f.dispLeft);
    d.right      = det.detect(f.grayRight, f.dispRight);
    d.bothValid  = d.left.found && d.right.found;
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// extractPoseMetrics
// ─────────────────────────────────────────────────────────────────────────────
static PoseMetrics extractPoseMetrics(const DetectionResult& det) {
    PoseMetrics pose;
    if (!det.found || det.corners.empty()) return pose;

    cv::Point2f sum(0, 0);
    for (const auto& p : det.corners) sum += p;
    pose.center = sum * (1.0f / det.corners.size());

    cv::Rect bbox      = cv::boundingRect(det.corners);
    pose.boardWidthPx  = (float)bbox.width;
    pose.boardHeightPx = (float)bbox.height;
    pose.rotationDeg   = cv::minAreaRect(det.corners).angle;
    pose.valid         = true;
    return pose;
}

// ─────────────────────────────────────────────────────────────────────────────
// updateCoverage
// ─────────────────────────────────────────────────────────────────────────────
static void updateCoverage(CoverageTracker& cov, const PoseMetrics& pose,
                           const cv::Size& imgSize) {
    if (!pose.valid) return;

    float x = pose.center.x / imgSize.width;
    float y = pose.center.y / imgSize.height;

    if      (x < 0.33f) cov.left   = true;
    else if (x > 0.66f) cov.right  = true;
    else                 cov.center = true;

    if (y < 0.33f) cov.top    = true;
    if (y > 0.66f) cov.bottom = true;

    float avg = (pose.boardWidthPx + pose.boardHeightPx) * 0.5f;
    if      (avg > 320) cov.near = true;
    else if (avg > 180) cov.mid  = true;
    else                cov.far  = true;

    float a = pose.rotationDeg;
    if      (a >  10.0f) cov.tiltRight = true;
    else if (a < -10.0f) cov.tiltLeft  = true;
    else                  cov.flat      = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// drawHUD — status overlay on a calibration frame
// ─────────────────────────────────────────────────────────────────────────────
static void drawHUD(cv::Mat& frame, bool bothValid, int done, int target, double fps) {
    // Valid/Invalid pill
    cv::Scalar clr = bothValid ? cv::Scalar(0,220,0) : cv::Scalar(0,50,220);
    cv::rectangle(frame, {10,8}, {bothValid ? 110 : 140, 48}, clr, cv::FILLED);
    cv::putText(frame, bothValid ? "VALID" : "INVALID", {18,38},
                cv::FONT_HERSHEY_SIMPLEX, 0.9, {255,255,255}, 2);

    // Pair counter
    cv::putText(frame, "Pairs: " + std::to_string(done) + " / " + std::to_string(target),
                {10,72}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {255,255,255}, 2);

    // FPS
    cv::putText(frame, "FPS: " + std::to_string((int)fps),
                {frame.cols - 110, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,255,255}, 1);

    // Bottom bar
    cv::rectangle(frame, {0, frame.rows-30}, {frame.cols, frame.rows}, {30,30,30}, cv::FILLED);
    cv::putText(frame, "SPACE: capture   ESC: exit",
                {10, frame.rows-9}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {200,200,200}, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// (Coverage Overlay Removed per user request to clean UI)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// renderCalibrationUI
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat renderCalibrationUI(StereoFrames& f, const StereoDetection& det,
                                   CalibrationSession& session,
                                   const CalibrationConfig& config,
                                   CameraStream& cam1, CameraStream& cam2,
                                   CoverageTracker& cov) {
    drawHUD(f.dispLeft,  det.bothValid, session.pairCount(), config.targetPairs, cam1.fps);
    drawHUD(f.dispRight, det.bothValid, session.pairCount(), config.targetPairs, cam2.fps);
    // Coverage overlay removed to keep UI clean

    cv::Mat combined;
    cv::hconcat(f.dispLeft, f.dispRight, combined);
    return combined;
}

// ─────────────────────────────────────────────────────────────────────────────
// flashCapture — green border flash on capture
// ─────────────────────────────────────────────────────────────────────────────
static void flashCapture(const cv::Mat& combined) {
    cv::Mat flash = combined.clone();
    cv::rectangle(flash, {0,0}, {flash.cols-1, flash.rows-1}, {0,255,0}, 10);
    cv::imshow("Stereo Calibration", flash);
    cv::waitKey(200);
}

// ─────────────────────────────────────────────────────────────────────────────
// handleCapture — triggered on SPACE key
// ─────────────────────────────────────────────────────────────────────────────
static void handleCapture(int key, const StereoDetection& det, StereoFrames& f,
                          CalibrationSession& session, const CalibrationConfig& config,
                          std::chrono::steady_clock::time_point& lastCapture,
                          const cv::Mat& ui) {
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::milliseconds;

    if (key != 32 || !det.bothValid) return;
    auto now = Clock::now();
    if (now - lastCapture < Ms(config.cooldownMs)) return;
    lastCapture = now;
    if (session.savePair(f.rawLeft, f.rawRight)) flashCapture(ui);
}

// ─────────────────────────────────────────────────────────────────────────────
// handleExit — triggered on ESC key
// ─────────────────────────────────────────────────────────────────────────────
static bool handleExit(int key, const CalibrationSession& session) {
    if (key != 27) return false;
    std::cout << "[CalibMode] Aborted. " << session.pairCount() << " pairs saved.\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// runCalibrationMode — public entry point
// ─────────────────────────────────────────────────────────────────────────────
void runCalibrationMode(CameraStream& cam1, CameraStream& cam2,
                        const CalibrationConfig& config) {
    ChessboardDetector detector(config);

    std::cout << "\n[N] New dataset\n[M] Append dataset\n";
    char choice;
    std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    const bool appendMode = (choice == 'm' || choice == 'M');

    if (!appendMode) {
        std::filesystem::remove_all(config.datasetDir + "/left");
        std::filesystem::remove_all(config.datasetDir + "/right");
        std::cout << "[CalibMode] Previous dataset removed.\n";
    }

    CalibrationSession session(config);
    CoverageTracker    coverage;
    const int          targetPairs = config.targetPairs;

    std::cout
        << "\n[CalibMode] ════════════════════════════════════\n"
        << "[CalibMode] Board : " << config.boardSize.width << "×" << config.boardSize.height << " inner corners\n"
        << "[CalibMode] Square: " << config.squareSizeM * 1000.0f << " mm\n"
        << "[CalibMode] Target: " << targetPairs << " pairs\n"
        << "[CalibMode] SPACE = capture | ESC = exit\n"
        << "[CalibMode] ════════════════════════════════════\n\n";

    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::milliseconds;
    auto lastCapture = Clock::now() - Ms(config.cooldownMs * 2);

    while (session.pairCount() < targetPairs) {
        StereoFrames frames = acquireFrames(cam1, cam2);
        if (!frames.valid()) {
            if (cv::waitKey(1) == 27) return;
            continue;
        }

        StereoDetection detection = detectStereoBoards(detector, frames);
        updateCoverage(coverage, extractPoseMetrics(detection.left), frames.rawLeft.size());

        cv::Mat combined = renderCalibrationUI(frames, detection, session, config,
                                               cam1, cam2, coverage);
        cv::imshow("Stereo Calibration", combined);

        int key = cv::waitKey(1);
        if (handleExit(key, session)) return;
        handleCapture(key, detection, frames, session, config, lastCapture, combined);
    }

    std::cout << "[CalibMode] Done! " << session.pairCount()
              << " pairs in '" << config.datasetDir << "'\n"
              << "[CalibMode] Next step: run stereoCalibrate on these pairs.\n";
}