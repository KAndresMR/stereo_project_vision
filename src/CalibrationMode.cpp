#include "CalibrationMode.hpp"
#include "ChessboardDetector.hpp"
#include "CalibrationSession.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers  (static = translation-unit scope, no header needed)
// ─────────────────────────────────────────────────────────────────────────────

// Thread-safe frame grab from a CameraStream.
static cv::Mat grabFrame(CameraStream& stream) {
    std::lock_guard<std::mutex> lock(stream.frame_mtx);
    return stream.frame.empty() ? cv::Mat{} : stream.frame.clone();
}

// Draw all HUD elements onto a single frame.
// Called once per side (left/right) before hconcat.
static void drawHUD(cv::Mat&            frame,
                    bool                bothValid,
                    int                 pairsDone,
                    int                 pairsTarget,
                    double              fps) {

    // ── Status pill (VALID / INVALID) ────────────────────────────────────────
    const std::string label    = bothValid ? "VALID"              : "INVALID";
    const cv::Scalar  labelClr = bothValid ? cv::Scalar(0,220,0) : cv::Scalar(0,50,220);

    // Filled rectangle behind the text for legibility
    cv::rectangle(frame,
                  cv::Point(10, 8),
                  cv::Point(bothValid ? 110 : 140, 48),
                  labelClr, cv::FILLED);
    cv::putText(frame, label,
                cv::Point(18, 38),
                cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(255, 255, 255), 2);

    // ── Pair counter ─────────────────────────────────────────────────────────
    std::string counter = "Pairs: " + std::to_string(pairsDone)
                        + " / "    + std::to_string(pairsTarget);
    cv::putText(frame, counter,
                cv::Point(10, 72),
                cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(255, 255, 255), 2);

    // ── FPS (top-right) ──────────────────────────────────────────────────────
    std::string fpsStr = "FPS: " + std::to_string(static_cast<int>(fps));
    cv::putText(frame, fpsStr,
                cv::Point(frame.cols - 110, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(0, 255, 255), 1);

    // ── Bottom hint bar ──────────────────────────────────────────────────────
    cv::rectangle(frame,
                  cv::Point(0, frame.rows - 30),
                  cv::Point(frame.cols, frame.rows),
                  cv::Scalar(30, 30, 30), cv::FILLED);
    cv::putText(frame, "SPACE: capture   ESC: exit",
                cv::Point(10, frame.rows - 9),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(200, 200, 200), 1);
}

// Brief green-flash feedback after a successful capture.
static void flashCapture(const cv::Mat& combined) {
    cv::Mat flash = combined.clone();
    cv::rectangle(flash,
                  cv::Point(0, 0),
                  cv::Point(flash.cols - 1, flash.rows - 1),
                  cv::Scalar(0, 255, 0), 10);
    cv::imshow("Stereo Calibration", flash);
    cv::waitKey(200);
}

// ─────────────────────────────────────────────────────────────────────────────
// runCalibrationMode
// ─────────────────────────────────────────────────────────────────────────────

void runCalibrationMode(CameraStream& cam1, CameraStream& cam2,
                        const CalibrationConfig& config) {

    ChessboardDetector detector(config);
    CalibrationSession session(config);

    std::cout << "\n[CalibMode] ════════════════════════════════\n";
    std::cout << "[CalibMode] Board : "
              << config.boardSize.width << "×" << config.boardSize.height
              << " inner corners\n";
    std::cout << "[CalibMode] Square: "
              << config.squareSizeM * 1000.0f << " mm\n";
    std::cout << "[CalibMode] Target: "
              << config.targetPairs << " pairs\n";
    std::cout << "[CalibMode] SPACE = capture | ESC = exit\n";
    std::cout << "[CalibMode] ════════════════════════════════\n\n";

    using Clock    = std::chrono::steady_clock;
    using Ms       = std::chrono::milliseconds;
    auto lastCapture = Clock::now() - Ms(config.cooldownMs * 2);

    while (!session.isComplete()) {

        // ── Grab latest frames from both streams ─────────────────────────────
        cv::Mat raw1 = grabFrame(cam1);
        cv::Mat raw2 = grabFrame(cam2);

        if (raw1.empty() || raw2.empty()) {
            if (cv::waitKey(1) == 27) return;
            continue;
        }

        // ── Convert to grayscale for detection ───────────────────────────────
        // Detection happens on gray; drawing happens on colour clones.
        cv::Mat gray1, gray2;
        cv::cvtColor(raw1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(raw2, gray2, cv::COLOR_BGR2GRAY);

        cv::Mat disp1 = raw1.clone();
        cv::Mat disp2 = raw2.clone();

        // ── Chessboard detection ──────────────────────────────────────────────
        DetectionResult r1 = detector.detect(gray1, disp1);
        DetectionResult r2 = detector.detect(gray2, disp2);

        bool bothValid = r1.found && r2.found;

        // ── HUD overlay ──────────────────────────────────────────────────────
        drawHUD(disp1, bothValid, session.pairCount(), config.targetPairs, cam1.fps);
        drawHUD(disp2, bothValid, session.pairCount(), config.targetPairs, cam2.fps);

        // ── Side-by-side window ───────────────────────────────────────────────
        cv::Mat combined;
        cv::hconcat(disp1, disp2, combined);
        cv::imshow("Stereo Calibration", combined);

        // ── Keyboard input ────────────────────────────────────────────────────
        int key = cv::waitKey(1);

        if (key == 27) {   // ESC
            std::cout << "[CalibMode] Aborted. "
                      << session.pairCount() << " pairs saved.\n";
            return;
        }

        if (key == 32 && bothValid) {   // SPACE — only when both cameras see the board
            auto now = Clock::now();
            if (now - lastCapture >= Ms(config.cooldownMs)) {
                lastCapture = now;
                // Save the RAW (colour, undistorted) frames.
                // Calibration algorithms need the original pixel values.
                if (session.savePair(raw1, raw2)) {
                    flashCapture(combined);
                }
            }
        }
    }

    std::cout << "[CalibMode] Done! "
              << session.pairCount() << " pairs in '"
              << config.datasetDir << "'\n";
    std::cout << "[CalibMode] Next step: run stereoCalibrate on these pairs.\n";
}
