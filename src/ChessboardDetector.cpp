#include "ChessboardDetector.hpp"
#include <iostream>

ChessboardDetector::ChessboardDetector(const CalibrationConfig& config)
    : config_(config) {}

DetectionResult ChessboardDetector::detect(const cv::Mat& grayFrame,
                                           cv::Mat&       displayFrame) const {
    DetectionResult result;

    // ── Step 1: coarse detection ──────────────────────────────────────────────
    //
    // findChessboardCorners scans the image for the chessboard pattern.
    // It works in pixel-level accuracy at this stage.
    //
    // The three flags passed via config_.findFlags:
    //   ADAPTIVE_THRESH  — uses adaptive thresholding instead of a global
    //                      threshold; handles non-uniform lighting across the
    //                      board (e.g. one corner in shadow).
    //   NORMALIZE_IMAGE  — stretches contrast before thresholding; helps when
    //                      the overall image is too dark or too bright.
    //   FAST_CHECK       — performs a cheap frequency-domain check first;
    //                      returns false immediately if the board is clearly
    //                      absent, saving ~15ms per frame when board is hidden.
    //
    result.found = cv::findChessboardCorners(
        grayFrame,
        config_.boardSize,
        result.corners,
        config_.findFlags
    );

    if (!result.found) {
        return result;   // nothing more to do; display frame is unchanged
    }

    // ── Step 2: sub-pixel refinement ─────────────────────────────────────────
    //
    // findChessboardCorners gives integer or at best ±0.5 px accuracy.
    // cornerSubPix iteratively moves each corner to the position where the
    // image gradient is orthogonal to the corner direction — this can
    // achieve ~0.01 px accuracy, which matters a lot for calibration quality.
    //
    // Search window (11,11) means a 23×23 px neighbourhood is analysed.
    // Zero zone (-1,-1) means no dead zone in the centre of the window.
    //
    cv::cornerSubPix(
        grayFrame,
        result.corners,
        config_.subPixWinSize,
        cv::Size(-1, -1),
        config_.subPixCriteria
    );

    // ── Step 3: draw corners on the display frame ────────────────────────────
    //
    // drawChessboardCorners colour-codes corners (green when pattern is
    // complete, red when partial) and connects them in order.
    //
    cv::drawChessboardCorners(
        displayFrame,
        config_.boardSize,
        result.corners,
        result.found
    );

    return result;
}
