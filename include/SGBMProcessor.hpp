#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <deque>
#include <mutex>

// ─────────────────────────────────────────────────────────────────────────────
// SGBMProcessor — Stereo Matching + WLS + Temporal Smoothing
//
// Ownership model:
//   - Owns the StereoSGBM (left matcher) and the right matcher.
//   - Owns the WLS filter.
//   - Owns the temporal smoothing buffer.
//   - Stateless with respect to frames — compute() is pure.
//     Exception: temporalSmooth() is stateful (history buffer).
//
// Thread safety:
//   - setParams() is NOT thread safe — call only from the display thread
//     when the processing thread is paused (or use a copy + swap pattern).
//   - compute(), postprocess(), normalize() ARE safe to call from a single
//     processing thread concurrently with the display thread reading results.
//
// WLS filter note:
//   WLS requires a "right disparity" computed by running SGBM in the
//   reverse direction (right→left). This costs ~2× CPU per frame but gives
//   significantly better edges and fills occlusion regions.
//   Controlled by the useWLS flag in Params.
// ─────────────────────────────────────────────────────────────────────────────
class SGBMProcessor {
public:

    struct Params {
        // ── SGBM ─────────────────────────────────────────────────────────────
        int minDisparity     = 16;
        int numDisparities   = 96;   // must be % 16 == 0
        int blockSize        = 5;   // must be odd

        // Smoothness penalties. 0 → auto-computed from blockSize.
        int P1               = 0;
        int P2               = 0;

        int disp12MaxDiff    = 1;
        int preFilterCap     = 63; // Defecto 31
        int uniquenessRatio  = 15;
        int speckleWindowSize = 150;
        int speckleRange     = 2;
        int mode             = cv::StereoSGBM::MODE_SGBM_3WAY; // Probar MODE_SGBM

        // ── WLS filter ────────────────────────────────────────────────────────
        bool   useWLS    = true;
        double wlsLambda = 8000.0;
        double wlsSigma  = 1.0;

        // ── Temporal smoothing ────────────────────────────────────────────────
        bool  useTemporalSmoothing = true;
        int   temporalWindow       = 4;    // frames to blend
        float temporalAlpha        = 0.6f; // EMA weight [0=frozen, 1=no smoothing]

        // ── Depth output ──────────────────────────────────────────────────────
        float minDepthM = 0.10f;
        float maxDepthM = 5.00f;

        // ── Preprocesamiento CLAHE ──────────────────────────────────────────
        double claheClipLimit = 4.0;
        int claheTileSize = 8;

        // ── Preprocesamiento Gaussian Blur ──────────────────────────────────────────
        double preBlurSigma = 0.8;
        
    };

    explicit SGBMProcessor(const Params& params);

    // Resets all matchers and WLS filter. Call after setParams().
    void setParams(const Params& p);
    const Params& getParams() const { return params_; }

    // ── Core pipeline steps ───────────────────────────────────────────────────

    // Step 1: Compute raw disparity (CV_16S, real_disp = value / 16.0).
    // rectLeft and rectRight must be already undistorted+rectified.
    // Internally applies CLAHE before SGBM for better contrast.
    cv::Mat compute(const cv::Mat& rectLeft, const cv::Mat& rectRight);

    // Step 2: WLS post-filter (fills holes, sharpens edges).
    // Pass the result of compute() and the raw rectLeft for edge guidance.
    // If useWLS==false, returns rawDisparity unchanged.
    cv::Mat postprocess(const cv::Mat& rawDisparity,
                        const cv::Mat& rectLeft);

    // Step 3: Temporal EMA smoothing across frames.
    // Reduces frame-to-frame flickering at the cost of slight lag.
    // If useTemporalSmoothing==false, returns input unchanged.
    cv::Mat temporalSmooth(const cv::Mat& filteredDisparity);

    // Step 4: Convert 16S disparity to 8-bit false-color for display.
    // COLORMAP_TURBO: blue=far, red=near.
    cv::Mat visualize(const cv::Mat& disparity) const;

    // ── Depth conversion ──────────────────────────────────────────────────────

    // Converts filtered disparity → metric depth (Z in meters) using Q matrix.
    // Invalid pixels (disp <= 0) and out-of-range depths are set to 0.
    // Returns CV_32F depth map.
    cv::Mat toDepth(const cv::Mat& disparity, const cv::Mat& Q) const;

    // Colorize depth map for display (near=red, far=blue).
    cv::Mat visualizeDepth(const cv::Mat& depthM) const;

    // Getter para obtener la imagen procesada con CLAHE y mostrarla en el dashboard
    cv::Mat getEnhancedLeft() const { return enhancedL_; }

private:
    Params params_;

    cv::Ptr<cv::StereoSGBM>                    leftMatcher_;
    cv::Ptr<cv::StereoMatcher>                  rightMatcher_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter>   wlsFilter_;
    cv::Ptr<cv::CLAHE>                          clahe_;

    // Temporal smoothing state (CV_32F accumulated average)
    cv::Mat  smoothed_;
    bool     smoothedValid_ = false;

    // Right disparity (needed by WLS, computed in compute() when useWLS=true)
    cv::Mat  rightDisp_;

    void init();  // (re)create all matchers from params_
    
    // Variable para guardar el resultado del CLAHE
    cv::Mat enhancedL_;
};