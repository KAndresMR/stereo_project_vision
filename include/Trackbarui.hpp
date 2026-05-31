#pragma once
#include <opencv2/opencv.hpp>
#include <mutex>
#include <atomic>
#include <string>
#include "SGBMProcessor.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// TrackbarUI
//
// Encapsulates all OpenCV trackbar logic for SGBM parameter tuning.
//
// Design decisions:
//
//   1. SINGLETON via static pointer.
//      OpenCV trackbar callbacks must be plain C functions or lambdas with
//      no capture. The only clean way to access class state from them is a
//      static pointer. One TrackbarUI per process — acceptable here.
//
//   2. INT→PARAM mapping happens at read time, not at write time.
//      Sliders store raw int values (OpenCV requirement). The conversion to
//      correct SGBM ranges (blockSize must be odd, numDisparities % 16 == 0)
//      happens in getParams(), not in the callback. This simplifies callbacks
//      to a single store operation.
//
//   3. Thread-safe via mutex + atomic flag.
//      The processing thread calls getParams() and hasChanged() concurrently
//      with the display thread driving OpenCV events. The mutex protects the
//      raw slider values; the atomic flag avoids the mutex for the hot path.
//
//   4. SGBM is not recreated on every slider drag.
//      hasChanged() returns true once and then resets. The caller recreates
//      SGBM only when the flag is set, not every frame.
//
// Usage:
//   TrackbarUI ui;
//   ui.create("SGBM Controls", initialParams);
//
//   while (running) {
//       cv::waitKey(1);                   // drives trackbar events
//       if (ui.hasChanged()) {
//           processor.setParams(ui.getParams());
//       }
//   }
// ─────────────────────────────────────────────────────────────────────────────
class TrackbarUI {
public:

    explicit TrackbarUI() = default;

    // Creates the named window and all trackbars.
    // Must be called from the main/display thread.
    void create(const std::string& windowName,
                const SGBMProcessor::Params& initial) {
        windowName_ = windowName;
        cv::namedWindow(windowName_, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName_, 400, 350);

        // Store initial raw slider values from params
        {
            std::lock_guard<std::mutex> l(mtx_);
            sliders_.blockSize        = (initial.blockSize - 1) / 2;   // bs=2n+1
            sliders_.numDisparities   = initial.numDisparities / 16;    // nd=16n
            sliders_.uniquenessRatio  = initial.uniquenessRatio;
            sliders_.speckleWinSize   = initial.speckleWindowSize;
            sliders_.speckleRange     = initial.speckleRange;
            sliders_.disp12MaxDiff    = initial.disp12MaxDiff;
            sliders_.preFilterCap     = initial.preFilterCap;
            sliders_.wlsLambda        = (int)(initial.wlsLambda / 100.0); // λ=100n
            sliders_.wlsSigma         = (int)(initial.wlsSigma * 10.0);   // σ=n/10
        }

        instance_ = this;

        // ── SGBM parameters ──────────────────────────────────────────────────
        cv::createTrackbar("blockSize (2n+1)", windowName_,
            &sliders_.blockSize, 10, cbAny, nullptr);

        cv::createTrackbar("numDisp (16n)", windowName_,
            &sliders_.numDisparities, 16, cbAny, nullptr);

        cv::createTrackbar("uniquenessRatio", windowName_,
            &sliders_.uniquenessRatio, 30, cbAny, nullptr);

        cv::createTrackbar("speckleWinSize", windowName_,
            &sliders_.speckleWinSize, 300, cbAny, nullptr);

        cv::createTrackbar("speckleRange", windowName_,
            &sliders_.speckleRange, 100, cbAny, nullptr);

        cv::createTrackbar("disp12MaxDiff", windowName_,
            &sliders_.disp12MaxDiff, 10, cbAny, nullptr);

        cv::createTrackbar("preFilterCap", windowName_,
            &sliders_.preFilterCap, 127, cbAny, nullptr);

        // ── WLS parameters ────────────────────────────────────────────────────
        cv::createTrackbar("WLS lambda (x100)", windowName_,
            &sliders_.wlsLambda, 200, cbAny, nullptr);

        cv::createTrackbar("WLS sigma (x10)", windowName_,
            &sliders_.wlsSigma, 30, cbAny, nullptr);

        changed_ = true;  // force initial apply
    }

    // Returns true once per change event, then resets.
    bool hasChanged() {
        return changed_.exchange(false);
    }

    // Thread-safe: converts raw slider ints to valid SGBM params.
    SGBMProcessor::Params getParams() const {
        std::lock_guard<std::mutex> l(mtx_);
        SGBMProcessor::Params p;

        // blockSize: slider n → 2n+1 (ensures odd, min 3)
        p.blockSize = std::max(3, sliders_.blockSize * 2 + 1);

        // numDisparities: slider n → 16n (must be multiple of 16, min 16)
        p.numDisparities = std::max(16, sliders_.numDisparities * 16);

        p.uniquenessRatio  = std::max(0, sliders_.uniquenessRatio);
        p.speckleWindowSize = sliders_.speckleWinSize;
        p.speckleRange     = sliders_.speckleRange;
        p.disp12MaxDiff    = sliders_.disp12MaxDiff;
        p.preFilterCap     = std::max(1, sliders_.preFilterCap);

        // WLS: slider n → λ = 100n, σ = n/10
        p.wlsLambda = sliders_.wlsLambda * 100.0;
        p.wlsSigma  = sliders_.wlsSigma  / 10.0;

        return p;
    }

    void destroy() {
        cv::destroyWindow(windowName_);
        instance_ = nullptr;
    }

private:
    std::string windowName_;
    mutable std::mutex mtx_;
    std::atomic<bool>  changed_{false};

    // Raw int values for OpenCV (sliders must be int*)
    struct Sliders {
        int blockSize        = 5;
        int numDisparities   = 4;
        int uniquenessRatio  = 10;
        int speckleWinSize   = 100;
        int speckleRange     = 32;
        int disp12MaxDiff    = 1;
        int preFilterCap     = 63;
        int wlsLambda        = 80;   // 8000 / 100
        int wlsSigma         = 15;   // 1.5 * 10
    } sliders_;

    // Singleton pointer for C-style callback
    static TrackbarUI* instance_;

    // Single callback for all sliders — just sets the changed flag.
    // The actual value is in the int* passed by OpenCV (already updated).
    static void cbAny(int, void*) {
        if (instance_) instance_->changed_ = true;
    }
};

// Definition of static member (put in ONE .cpp that includes this header)
inline TrackbarUI* TrackbarUI::instance_ = nullptr;