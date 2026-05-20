#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// CalibrationConfig — single source of truth for all pipeline parameters.
//
// Organized by phase so you know exactly which values affect which step.
// ─────────────────────────────────────────────────────────────────────────────
struct CalibrationConfig {

    // ── Chessboard ───────────────────────────────────────────────────────────
    cv::Size boardSize{9, 6};     // inner corners (not squares)
    float    squareSizeM = 0.025f; // physical square size in METERS (25mm)

    // ── Session ──────────────────────────────────────────────────────────────
    int         targetPairs = 30;
    std::string datasetDir  = "calib_pairs";   // root for left/ and right/ images

    // ── Output YAMLs ─────────────────────────────────────────────────────────
    // Phase 1 — individual calibration
    std::string leftYaml  = "calib_pairs/calib/left.yaml";
    std::string rightYaml = "calib_pairs/calib/right.yaml";
    // Phase 2 — stereo calibration
    std::string stereoYaml = "calib_pairs/calib/stereo.yaml";

    // ── Detection flags ──────────────────────────────────────────────────────
    int findFlags = cv::CALIB_CB_ADAPTIVE_THRESH
                  | cv::CALIB_CB_NORMALIZE_IMAGE
                  | cv::CALIB_CB_FAST_CHECK;

    // ── Sub-pixel refinement ─────────────────────────────────────────────────
    cv::TermCriteria subPixCriteria{
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
        30, 0.001
    };
    cv::Size subPixWinSize{11, 11};

    // ── CLAHE (illumination compensation) ────────────────────────────────────
    // When the mean frame brightness drops below brightnessThreshold,
    // CLAHE is applied before corner detection to recover contrast.
    // Safe to leave on — CLAHE is a no-op on well-lit frames.
    bool   autoEnhance        = true;
    float  brightnessThreshold = 80.0f;  // 0–255 mean pixel value
    double claheClipLimit      = 2.0;    // higher = more aggressive
    cv::Size claheTileSize{8, 8};

    // ── Capture ───────────────────────────────────────────────────────────────
    int cooldownMs = 800;  // min ms between captures (prevents double-trigger)

    // ── Stereo calibration flags ─────────────────────────────────────────────
    // FIX_INTRINSIC: use K/dist from Phase 1 as-is, only optimize R/T.
    // This is the correct approach when individual calibration was good.
    int stereoFlags = cv::CALIB_FIX_INTRINSIC;
};