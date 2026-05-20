#include "ChessboardDetector.hpp"
#include "DiagnosticLogger.hpp"

ChessboardDetector::ChessboardDetector(const CalibrationConfig& config)
    : config_(config) {}

// ── Private helpers ──────────────────────────────────────────────────────────

double ChessboardDetector::measureBrightness(const cv::Mat& gray) const {
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    return mean[0];  // mean pixel intensity, range 0–255
}

double ChessboardDetector::measureContrast(const cv::Mat& gray) const {
    // Michelson contrast: (max - min) / (max + min)
    // Range 0 (flat gray) to 1 (pure black and white)
    double minVal, maxVal;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    if (maxVal + minVal < 1.0) return 0.0;
    return (maxVal - minVal) / (maxVal + minVal);
}

cv::Mat ChessboardDetector::applyClahe(const cv::Mat& gray) const {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        config_.claheClipLimit,
        config_.claheTileSize
    );
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);
    return enhanced;
}

// ── Public detect ────────────────────────────────────────────────────────────

DetectionResult ChessboardDetector::detect(const cv::Mat& grayFrame,
                                           cv::Mat&       displayFrame) const {
    DetectionResult result;
    result.cornersExpected = config_.boardSize.width * config_.boardSize.height;

    // ── Step 1: measure image quality ────────────────────────────────────────
    result.brightness = measureBrightness(grayFrame);
    result.contrast   = measureContrast(grayFrame);

    // ── Step 2: decide whether to apply CLAHE ────────────────────────────────
    // CLAHE (Contrast Limited Adaptive Histogram Equalization) equalizes
    // contrast locally per tile. Helps when the chessboard is dark but
    // the pattern is still geometrically present.
    //
    // We work on a COPY — never modify the caller's grayFrame,
    // because the caller may need the original for saving to disk.
    cv::Mat workGray = grayFrame;
    if (config_.autoEnhance && result.brightness < config_.brightnessThreshold) {
        workGray         = applyClahe(grayFrame);
        result.claheUsed = true;
    }

    // ── Step 3: coarse detection ──────────────────────────────────────────────
    result.found = cv::findChessboardCorners(
        workGray,
        config_.boardSize,
        result.corners,
        config_.findFlags
    );

    // Count how many corners were found even if the board wasn't complete.
    // Useful for diagnosing partial occlusion vs total failure.
    result.cornersFound = static_cast<int>(result.corners.size());

    if (!result.found) {
        // If CLAHE was NOT tried yet and brightness is marginal (50–80),
        // try once more with CLAHE forced on.
        if (!result.claheUsed && result.brightness < 100.0) {
            cv::Mat enhanced = applyClahe(grayFrame);
            result.found = cv::findChessboardCorners(
                enhanced,
                config_.boardSize,
                result.corners,
                config_.findFlags
            );
            result.claheUsed    = true;
            result.cornersFound = static_cast<int>(result.corners.size());
        }

        if (!result.found) {
            return result;  // genuinely not found — caller will log via Log::detection
        }
    }

    // ── Step 4: sub-pixel refinement ─────────────────────────────────────────
    // cornerSubPix refines each corner from ~1px accuracy to ~0.01px.
    // Works on the same workGray used for detection (CLAHE-enhanced or not).
    cv::cornerSubPix(
        workGray,
        result.corners,
        config_.subPixWinSize,
        cv::Size(-1, -1),
        config_.subPixCriteria
    );

    // ── Step 5: draw on display frame ────────────────────────────────────────
    // Draw on the COLOUR display frame only — never on grayFrame or workGray.
    cv::drawChessboardCorners(
        displayFrame,
        config_.boardSize,
        result.corners,
        result.found
    );

    return result;
}