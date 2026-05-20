#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// CalibrationConfig
//
// Single source of truth for every tunable parameter in the calibration phase.
// Pass this struct by const-ref throughout the system — never hardcode values
// inside classes. This makes it trivial to change the board or square size.
// ─────────────────────────────────────────────────────────────────────────────
struct CalibrationConfig {

    // ── Chessboard ───────────────────────────────────────────────────────────
    // INNER corners (squares - 1).  A standard 10×7 printed board has 9×6 here.
    cv::Size boardSize{9, 6};

    // Physical edge length of one square, in METERS.
    // Used later by stereoCalibrate to produce R/T in real-world units.
    float squareSizeM = 0.025f;   // 25 mm

    // ── Session ──────────────────────────────────────────────────────────────
    // How many valid stereo pairs to collect before declaring "done".
    int targetPairs = 30;

    // Root folder. Sub-folders left/ and right/ are created automatically.
    std::string outputDir = "calib_pairs";

    // ── cornerSubPix ─────────────────────────────────────────────────────────
    // Stopping criteria for sub-pixel corner refinement.
    // MAX_ITER + EPS: stop after 30 iterations OR when movement < 0.001 px.
    cv::TermCriteria subPixCriteria{
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
        30,
        0.001
    };

    // Search window half-size for cornerSubPix.
    // (11,11) → 23×23 pixel window — good default for VGA resolution.
    cv::Size subPixWinSize{11, 11};

    // ── Detection flags ──────────────────────────────────────────────────────
    // ADAPTIVE_THRESH: handles uneven lighting across the board.
    // NORMALIZE_IMAGE: normalises brightness before thresholding.
    // FAST_CHECK:      cheap early-reject when board is not visible.
    int findFlags = cv::CALIB_CB_ADAPTIVE_THRESH
                  | cv::CALIB_CB_NORMALIZE_IMAGE
                  | cv::CALIB_CB_FAST_CHECK;

    // ── Capture cooldown ─────────────────────────────────────────────────────
    // Minimum milliseconds between two consecutive captures (prevents
    // accidental double-captures from a single SPACE key press).
    int cooldownMs = 800;
};
