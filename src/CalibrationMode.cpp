#include "CalibrationMode.hpp"
#include "ChessboardDetector.hpp"
#include "CalibrationSession.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <chrono>
#include <iostream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Internal Types
// ─────────────────────────────────────────────────────────────────────────────

struct StereoFrames {
    cv::Mat rawLeft;
    cv::Mat rawRight;

    cv::Mat grayLeft;
    cv::Mat grayRight;

    cv::Mat dispLeft;
    cv::Mat dispRight;

    bool valid() const {
        return !rawLeft.empty() && !rawRight.empty();
    }
};

struct StereoDetection {
    DetectionResult left;
    DetectionResult right;

    bool bothValid = false;
};

struct PoseMetrics {

    cv::Point2f center;

    float boardWidthPx  = 0.0f;
    float boardHeightPx = 0.0f;

    float rotationDeg = 0.0f;

    bool valid = false;
};

struct CoverageTracker {

    // ── Spatial coverage ────────────────────────────────────────────────────
    bool top    = false;
    bool bottom = false;
    bool left   = false;
    bool right  = false;
    bool center = false;

    // ── Scale coverage ──────────────────────────────────────────────────────
    bool near  = false;
    bool mid   = false;
    bool far   = false;

    // ── Rotation coverage ───────────────────────────────────────────────────
    bool tiltLeft  = false;
    bool tiltRight = false;
    bool flat      = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe frame grab
// ─────────────────────────────────────────────────────────────────────────────

static cv::Mat grabFrame(CameraStream& stream) {
    std::lock_guard<std::mutex> lock(stream.frame_mtx);

    return stream.frame.empty()
        ? cv::Mat{}
        : stream.frame.clone();
}

// ─────────────────────────────────────────────────────────────────────────────
// Acquire stereo frames
// ─────────────────────────────────────────────────────────────────────────────

static StereoFrames acquireFrames(
    CameraStream& cam1,
    CameraStream& cam2)
{
    StereoFrames frames;

    frames.rawLeft  = grabFrame(cam1);
    frames.rawRight = grabFrame(cam2);

    if (!frames.valid())
        return frames;

    // ── Convert to grayscale ────────────────────────────────────────────────
    cv::cvtColor(
        frames.rawLeft,
        frames.grayLeft,
        cv::COLOR_BGR2GRAY
    );

    cv::cvtColor(
        frames.rawRight,
        frames.grayRight,
        cv::COLOR_BGR2GRAY
    );

    // ── Create display buffers ──────────────────────────────────────────────
    frames.dispLeft  = frames.rawLeft.clone();
    frames.dispRight = frames.rawRight.clone();

    return frames;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chessboard detection
// ─────────────────────────────────────────────────────────────────────────────

static StereoDetection detectStereoBoards(
    ChessboardDetector& detector,
    StereoFrames& frames)
{
    StereoDetection detection;

    detection.left =
        detector.detect(
            frames.grayLeft,
            frames.dispLeft
        );

    detection.right =
        detector.detect(
            frames.grayRight,
            frames.dispRight
        );

    detection.bothValid =
        detection.left.found &&
        detection.right.found;

    return detection;
}

static PoseMetrics extractPoseMetrics(
    const DetectionResult& detection)
{
    PoseMetrics pose;

    if (!detection.found || detection.corners.empty())
        return pose;

    // ── Compute center ──────────────────────────────────────────────────────
    cv::Point2f sum(0,0);

    for (const auto& p : detection.corners)
        sum += p;

    pose.center =
        sum * (1.0f / detection.corners.size());

    // ── Bounding box ────────────────────────────────────────────────────────
    cv::Rect bbox =
        cv::boundingRect(detection.corners);

    pose.boardWidthPx  = static_cast<float>(bbox.width);
    pose.boardHeightPx = static_cast<float>(bbox.height);

    // ── Approximate rotation ────────────────────────────────────────────────
    cv::RotatedRect rect =
        cv::minAreaRect(detection.corners);

    pose.rotationDeg = rect.angle;

    pose.valid = true;

    return pose;
}

static void updateCoverage(
    CoverageTracker& coverage,
    const PoseMetrics& pose,
    const cv::Size& imageSize)
{
    if (!pose.valid)
        return;

    float x = pose.center.x / imageSize.width;
    float y = pose.center.y / imageSize.height;

    // ── Spatial regions ─────────────────────────────────────────────────────
    if (x < 0.33f)
        coverage.left = true;

    else if (x > 0.66f)
        coverage.right = true;

    else
        coverage.center = true;

    if (y < 0.33f)
        coverage.top = true;

    if (y > 0.66f)
        coverage.bottom = true;

    // ── Scale estimation ────────────────────────────────────────────────────
    float avgSize =
        (pose.boardWidthPx + pose.boardHeightPx) * 0.5f;

    if (avgSize > 320)
        coverage.near = true;

    else if (avgSize > 180)
        coverage.mid = true;

    else
        coverage.far = true;

    // ── Rotation estimation ─────────────────────────────────────────────────
    float a = pose.rotationDeg;

    if (a > 10.0f)
        coverage.tiltRight = true;

    else if (a < -10.0f)
        coverage.tiltLeft = true;

    else
        coverage.flat = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Draw HUD
// ─────────────────────────────────────────────────────────────────────────────

static void drawHUD(
    cv::Mat& frame,
    bool bothValid,
    int pairsDone,
    int pairsTarget,
    double fps)
{
    // ── VALID / INVALID pill ────────────────────────────────────────────────
    const std::string label =
        bothValid ? "VALID" : "INVALID";

    const cv::Scalar labelClr =
        bothValid
            ? cv::Scalar(0,220,0)
            : cv::Scalar(0,50,220);

    cv::rectangle(
        frame,
        cv::Point(10, 8),
        cv::Point(bothValid ? 110 : 140, 48),
        labelClr,
        cv::FILLED
    );

    cv::putText(
        frame,
        label,
        cv::Point(18, 38),
        cv::FONT_HERSHEY_SIMPLEX,
        0.9,
        cv::Scalar(255,255,255),
        2
    );

    // ── Pair counter ────────────────────────────────────────────────────────
    std::string counter =
        "Pairs: " +
        std::to_string(pairsDone) +
        " / " +
        std::to_string(pairsTarget);

    cv::putText(
        frame,
        counter,
        cv::Point(10, 72),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(255,255,255),
        2
    );

    // ── FPS ─────────────────────────────────────────────────────────────────
    std::string fpsStr =
        "FPS: " +
        std::to_string(static_cast<int>(fps));

    cv::putText(
        frame,
        fpsStr,
        cv::Point(frame.cols - 110, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0,255,255),
        1
    );

    // ── Bottom bar ──────────────────────────────────────────────────────────
    cv::rectangle(
        frame,
        cv::Point(0, frame.rows - 30),
        cv::Point(frame.cols, frame.rows),
        cv::Scalar(30,30,30),
        cv::FILLED
    );

    cv::putText(
        frame,
        "SPACE: capture   ESC: exit",
        cv::Point(10, frame.rows - 9),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(200,200,200),
        1
    );
}

static void drawCoverageOverlay(
    cv::Mat& frame,
    const CoverageTracker& c)
{
    int x = 10;
    int y = 110;

    auto mark = [](bool ok) -> std::string {
        return ok ? "[✓] " : "[ ] ";
    };

    std::vector<std::string> lines = {

        "Coverage:",

        mark(c.center) + "Center",
        mark(c.top)    + "Top",
        mark(c.bottom) + "Bottom",
        mark(c.left)   + "Left",
        mark(c.right)  + "Right",

        "",

        mark(c.near) + "Near",
        mark(c.mid)  + "Medium",
        mark(c.far)  + "Far",

        "",

        mark(c.flat)      + "Flat",
        mark(c.tiltLeft)  + "Tilt Left",
        mark(c.tiltRight) + "Tilt Right",
    };

    for (size_t i = 0; i < lines.size(); ++i) {

        cv::putText(
            frame,
            lines[i],
            cv::Point(x, y + i * 18),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255,255,255),
            1
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render calibration UI
// ─────────────────────────────────────────────────────────────────────────────

static cv::Mat renderCalibrationUI(
    StereoFrames& frames,
    const StereoDetection& detection,
    CalibrationSession& session,
    const CalibrationConfig& config,
    CameraStream& cam1,
    CameraStream& cam2,
    CoverageTracker& coverage)
{
    drawHUD(
        frames.dispLeft,
        detection.bothValid,
        session.pairCount(),
        config.targetPairs,
        cam1.fps
    );

    drawHUD(
        frames.dispRight,
        detection.bothValid,
        session.pairCount(),
        config.targetPairs,
        cam2.fps
    );

    drawCoverageOverlay(
        frames.dispLeft,
        coverage
    );

    cv::Mat combined;

    cv::hconcat(
        frames.dispLeft,
        frames.dispRight,
        combined
    );

    return combined;
}

// ─────────────────────────────────────────────────────────────────────────────
// Flash capture feedback
// ─────────────────────────────────────────────────────────────────────────────

static void flashCapture(const cv::Mat& combined) {

    cv::Mat flash = combined.clone();

    cv::rectangle(
        flash,
        cv::Point(0, 0),
        cv::Point(flash.cols - 1, flash.rows - 1),
        cv::Scalar(0,255,0),
        10
    );

    cv::imshow("Stereo Calibration", flash);

    cv::waitKey(200);
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle capture
// ─────────────────────────────────────────────────────────────────────────────

static void handleCapture(
    int key,
    const StereoDetection& detection,
    StereoFrames& frames,
    CalibrationSession& session,
    const CalibrationConfig& config,
    std::chrono::steady_clock::time_point& lastCapture,
    const cv::Mat& currentUI)
{
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::milliseconds;

    // SPACE
    if (key != 32)
        return;

    if (!detection.bothValid)
        return;

    auto now = Clock::now();

    if (now - lastCapture < Ms(config.cooldownMs))
        return;

    lastCapture = now;

    if (session.savePair(
            frames.rawLeft,
            frames.rawRight))
    {
        flashCapture(currentUI);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle exit
// ─────────────────────────────────────────────────────────────────────────────

static bool handleExit(
    int key,
    CalibrationSession& session)
{
    if (key != 27)
        return false;

    std::cout
        << "[CalibMode] Aborted. "
        << session.pairCount()
        << " pairs saved.\n";

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main calibration mode
// ─────────────────────────────────────────────────────────────────────────────

void runCalibrationMode(
    CameraStream& cam1,
    CameraStream& cam2,
    const CalibrationConfig& config)
{
    ChessboardDetector detector(config);

    std::cout << "\n";
    std::cout << "[N] New dataset\n";
    std::cout << "[M] Append dataset\n";

    char choice;
    std::cin >> choice;

    bool appendMode =
        (choice == 'm' || choice == 'M');

    if (!appendMode) {

        std::filesystem::remove_all(
            config.datasetDir + "/left");

        std::filesystem::remove_all(
            config.datasetDir + "/right");

        std::cout
            << "[CalibMode] Previous dataset removed.\n";
    }
    CalibrationSession session(config);
    CoverageTracker coverage;

    int targetPairs = config.targetPairs;

    if (appendMode)
    {
        targetPairs += session.pairCount();
    }

    std::cout << "\n[CalibMode] ════════════════════════════════\n";

    std::cout << "[CalibMode] Target: "
            << targetPairs
            << " pairs\n";


    std::cout << "\n[CalibMode] ════════════════════════════════\n";
    std::cout << "[CalibMode] Board : "
              << config.boardSize.width
              << "×"
              << config.boardSize.height
              << " inner corners\n";

    std::cout << "[CalibMode] Square: "
              << config.squareSizeM * 1000.0f
              << " mm\n";

    std::cout << "[CalibMode] Target: "
              << targetPairs
              << " pairs\n";

    std::cout << "[CalibMode] SPACE = capture | ESC = exit\n";

    std::cout << "[CalibMode] ════════════════════════════════\n\n";

    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::milliseconds;

    auto lastCapture =
        Clock::now() - Ms(config.cooldownMs * 2);

    while (session.pairCount() < targetPairs) {

        // ── Acquire frames ──────────────────────────────────────────────────
        StereoFrames frames =
            acquireFrames(cam1, cam2);

        if (!frames.valid()) {

            if (cv::waitKey(1) == 27)
                return;

            continue;
        }

        // ── Detect chessboards ──────────────────────────────────────────────
        StereoDetection detection =
            detectStereoBoards(detector, frames);

        PoseMetrics pose =
            extractPoseMetrics(detection.left);

        updateCoverage(
            coverage,
            pose,
            frames.rawLeft.size()
        );

        // ── Render UI ───────────────────────────────────────────────────────
        cv::Mat combined =
            renderCalibrationUI(
                frames,
                detection,
                session,
                config,
                cam1,
                cam2,
                coverage
            );

        cv::imshow(
            "Stereo Calibration",
            combined
        );

        // ── Input ───────────────────────────────────────────────────────────
        int key = cv::waitKey(1);

        if (handleExit(key, session))
            return;

        // ── Capture ─────────────────────────────────────────────────────────
        handleCapture(
            key,
            detection,
            frames,
            session,
            config,
            lastCapture,
            combined
        );
    }

    std::cout
        << "[CalibMode] Done! "
        << session.pairCount()
        << " pairs in '"
        << config.datasetDir
        << "'\n";

    std::cout
        << "[CalibMode] Next step: run stereoCalibrate on these pairs.\n";
}