#include "SGBMProcessor.hpp"
#include "DiagnosticLogger.hpp"
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
SGBMProcessor::SGBMProcessor(const Params& params) : params_(params) {
    init();
}

// ─────────────────────────────────────────────────────────────────────────────
// init — (re)create matchers from current params_
//
// Called at construction and after setParams().
// Expensive (~1ms) but not per-frame — only on parameter change.
// ─────────────────────────────────────────────────────────────────────────────
void SGBMProcessor::init() {
    const int ch = 1;
    const int bs = params_.blockSize;

    int p1 = (params_.P1 > 0) ? params_.P1 : 1200;
    int p2 = (params_.P2 > 0) ? params_.P2 : 5000;

    leftMatcher_ = cv::StereoSGBM::create(
        params_.minDisparity,
        params_.numDisparities,
        params_.blockSize,
        p1, p2,
        params_.disp12MaxDiff,
        params_.preFilterCap,
        params_.uniquenessRatio,
        params_.speckleWindowSize,
        params_.speckleRange,
        params_.mode
    );

    // Right matcher: mirrors the left matcher in the right→left direction.
    // Required for WLS filter. Cheap to create.
    if (params_.useWLS) {
        rightMatcher_ = cv::ximgproc::createRightMatcher(leftMatcher_);
        wlsFilter_    = cv::ximgproc::createDisparityWLSFilter(leftMatcher_);
        wlsFilter_->setLambda(params_.wlsLambda);
        wlsFilter_->setSigmaColor(params_.wlsSigma);
        Log::info("SGBM", "WLS filter enabled (λ=" +
                  std::to_string((int)params_.wlsLambda) +
                  " σ=" + std::to_string(params_.wlsSigma) + ")");
    }

    // CLAHE for pre-processing: improves SGBM on low-contrast scenes.
    // clip=2.0, tile=8×8 is a conservative setting — reduces halos.
    clahe_ = cv::createCLAHE(5.0, cv::Size(8, 8)); // Un Clip Limit de 2.0 es conservador y funciona genial con luz de día. En un cuarto oscuro con una cámara económica, 2.0 no hace casi nada.

    // Reset temporal buffer (params may have changed window size)
    smoothed_      = cv::Mat{};
    smoothedValid_ = false;

    Log::info("SGBM", "Initialized: numDisp=" +
              std::to_string(params_.numDisparities) +
              " blockSize=" + std::to_string(params_.blockSize) +
              " P1=" + std::to_string(p1) +
              " P2=" + std::to_string(p2));
}

// ─────────────────────────────────────────────────────────────────────────────
void SGBMProcessor::setParams(const Params& p) {
    params_ = p;
    init();
}

// ─────────────────────────────────────────────────────────────────────────────
// compute — Step 1
//
// Applies CLAHE to rectified grayscale images, then runs StereoSGBM.
// If WLS is enabled, also runs the right matcher here so postprocess()
// can use rightDisp_ without recomputing.
//
// Output: CV_16S disparity map. Real disparity = value / 16.0 pixels.
// Invalid pixels have value < params_.minDisparity * 16.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::compute(const cv::Mat& rectLeft, const cv::Mat& rectRight) {
    // Convert to grayscale (SGBM works on grayscale internally anyway,
    // but accepting colour input is more flexible for the caller)
    cv::Mat grayL, grayR;
    if (rectLeft.channels() == 3) {
        cv::cvtColor(rectLeft,  grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rectRight, grayR, cv::COLOR_BGR2GRAY);
    } else {
        grayL = rectLeft;
        grayR = rectRight;
    }

    // CLAHE — equalizes contrast locally per tile.
    // Helps SGBM find matches in dark or overexposed regions.
    // Actualizar CLAHE con los parámetros en vivo
    clahe_->setClipLimit(params_.claheClipLimit);
    
    // Evitar que el Tile Size sea 0 o negativo
    int ts = std::max(2, params_.claheTileSize);
    clahe_->setTilesGridSize(cv::Size(ts, ts));

    cv::Mat enhL, enhR;
    if(params_.preBlurSigma > 0.1){
        cv::GaussianBlur(grayL, grayL, cv::Size(3, 3), params_.preBlurSigma);
        cv::GaussianBlur(grayR, grayR, cv::Size(3, 3), params_.preBlurSigma);
    }
    clahe_->apply(grayL, enhL);
    clahe_->apply(grayR, enhR);

    // [!] NUEVO: Guardamos la imagen mejorada para usarla como guía y en el dashboard
    enhancedL_ = enhL.clone();

    // Left disparity (primary output)
    cv::Mat leftDisp;
    leftMatcher_->compute(enhL, enhR, leftDisp);

    // Right disparity (only needed for WLS; store for postprocess())
    if (params_.useWLS && rightMatcher_) {
        rightMatcher_->compute(enhR, enhL, rightDisp_);
    }

    return leftDisp;
}

// ─────────────────────────────────────────────────────────────────────────────
// postprocess — Step 2: WLS filter
//
// The WLS (Weighted Least Squares) filter solves:
//
//   argmin_u { ||u - d||² + λ * Σ w_ij(I) * (u_i - u_j)² }
//
// where d = raw disparity, w_ij(I) = edge-aware weight from reference image I.
//
// In plain language:
//   - It fills holes (where SGBM found no match) by diffusing neighboring
//     values, guided by image edges (doesn't blur across real depth edges).
//   - λ controls smoothness: higher → smoother but less detail.
//   - σ controls edge sensitivity: higher → diffuses more across edges.
//
// Why right disparity?
//   WLS uses the right disparity to compute a confidence map: a pixel is
//   "confident" if left→right and right→left disparities agree.
//   Unconfident pixels are filled by the filter, not passed through.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::postprocess(const cv::Mat& rawDisparity,
                                    const cv::Mat& rectLeft) {
    if (!params_.useWLS || !wlsFilter_ || rightDisp_.empty()) {
        return rawDisparity;
    }
    cv::Mat filtered;
    wlsFilter_->filter(
        rawDisparity,
        enhancedL_, // Usamos la imagen con CLAHE como guía
        filtered,
        rightDisp_
    );
    return filtered;
}

// ─────────────────────────────────────────────────────────────────────────────
// temporalSmooth — Step 3: Exponential Moving Average
//
// EMA formula: S(t) = α * D(t) + (1-α) * S(t-1)
//
// α = temporalAlpha:
//   - High (0.8–1.0): fast response, minimal smoothing.
//   - Low  (0.1–0.3): heavy smoothing, ghosting on moving objects.
//   - 0.4 is a good default for a mostly static scene.
//
// Works on CV_32F to avoid rounding errors in accumulation.
// Pixels that were invalid (<=0) in current frame are NOT updated,
// preserving the last valid estimate — avoids smearing invalid regions.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::temporalSmooth(const cv::Mat& filteredDisparity) {
    if (!params_.useTemporalSmoothing) return filteredDisparity;

    cv::Mat current32f;
    filteredDisparity.convertTo(current32f, CV_32F);

    if (!smoothedValid_ || smoothed_.size() != current32f.size()) {
        smoothed_      = current32f.clone();
        smoothedValid_ = true;
        return filteredDisparity;
    }

    // Only update pixels that have a valid disparity in the current frame.
    // Invalid = disparity <= minDisparity * 16 (SGBM convention).
    cv::Mat validMask = (filteredDisparity > params_.minDisparity * 16);

    // EMA on valid pixels only
    float alpha = params_.temporalAlpha;
    cv::Mat blended;

    cv::addWeighted(
        current32f,
        alpha,
        smoothed_,
        1.0f - alpha,
        0.0,
        blended);

    // Update ONLY valid pixels
    blended.copyTo(smoothed_, validMask);

    // Write back to invalid pixels (keep last valid estimate there)
    // — this is the "temporal infill" behavior
    cv::Mat result;
    smoothed_.convertTo(result, CV_16S);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// visualize — Step 4: 16S disparity → 8-bit false-color
//
// SGBM outputs CV_16S where each value = real_disparity * 16.
// We divide by 16, clip negative (invalid), scale to 0–255.
// COLORMAP_TURBO: warm colors = near, cool colors = far.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::visualize(const cv::Mat& disparity) const {
    // Convert to float, divide by 16 to get real disparity in pixels
    cv::Mat disp32f;
    disparity.convertTo(disp32f, CV_32F, 1.0 / 16.0);

    // Clip negatives (invalid regions)
    cv::threshold(disp32f, disp32f, 0.0, 0.0, cv::THRESH_TOZERO);

    // Normalize to 0–255
    double maxDisp = params_.numDisparities;
    cv::Mat disp8u;
    disp32f.convertTo(disp8u, CV_8U, 255.0 / maxDisp);

    // Apply color map
    //cv::Mat colored;
    //cv::applyColorMap(disp8u, colored, cv::COLORMAP_TURBO);

    // Black out invalid pixels
    cv::Mat invalid = (disparity <= params_.minDisparity * 16);
    //colored.setTo(cv::Scalar(0, 0, 0), invalid);
    disp8u.setTo(0, invalid);

    return disp8u;
}

// ─────────────────────────────────────────────────────────────────────────────
// toDepth — Disparity → real depth in meters via Q matrix
//
// The Q matrix from stereoRectify encodes:
//   Z = f * B / d
// where f = focal length, B = baseline, d = disparity.
//
// reprojectImageTo3D computes (X, Y, Z) for every pixel.
// We extract only Z (depth) and apply range clipping.
//
// Invalid pixels: disparity <= 0 after conversion → set depth = 0.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::toDepth(const cv::Mat& disparity,
                                const cv::Mat& Q) const {
    // Convert disparity to float (real disparity in pixels)
    cv::Mat disp32f;
    disparity.convertTo(disp32f, CV_32F, 1.0 / 16.0);

    // reprojectImageTo3D: for each pixel, compute (X, Y, Z) in camera coords
    // handleMissingValues=true: sets out-of-range pixels to 10000
    cv::Mat points3D;
    cv::reprojectImageTo3D(disp32f, points3D, Q, true);

    // Extract Z channel (depth in meters)
    cv::Mat channels[3];
    cv::split(points3D, channels);
    cv::Mat depthM = channels[2];

    // Mask: valid disparity AND depth in expected range
    cv::Mat validDisp = (disp32f > 0.0f);
    cv::Mat validDepth = (depthM > params_.minDepthM) &
                         (depthM < params_.maxDepthM);
    cv::Mat valid = validDisp & validDepth;

    static int counter = 0;

    if (++counter % 30 == 0) {
        std::cout << depthM.at<float>(240,320) << std::endl;
    }
    // Zero out invalid pixels
    depthM.setTo(0.0f, ~valid);

    return depthM;  // CV_32F, meters
}

// ─────────────────────────────────────────────────────────────────────────────
// visualizeDepth — depth map → false-color image for display
//
// Linear scale: minDepthM → blue (far), maxDepthM → red (near).
// Invalid (zero) pixels shown as black.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::visualizeDepth(const cv::Mat& depthM) const {
    cv::Mat normalized;
    double scale = 255.0 / (params_.maxDepthM - params_.minDepthM);
    depthM.convertTo(normalized, CV_8U, scale,
                     -params_.minDepthM * scale);

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);

    // Black out invalid pixels (depth == 0)
    colored.setTo(cv::Scalar(0,0,0), depthM == 0.0f);

    return colored;
}