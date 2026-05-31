#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include "CameraStream.hpp"
#include "Rectifier.hpp"
#include "SGBMProcessor.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// LivePipeline — Real-time stereo processing thread
//
// Threading model:
//
//   [Camera Thread L] ──→ CameraStream::frame (mutex protected)
//   [Camera Thread R] ──→ CameraStream::frame (mutex protected)
//                                   │
//                                   ▼  (processing thread grabs LATEST pair)
//   [Processing Thread] ──→ rectify → SGBM → depth → Result (single slot)
//                                   │
//                                   ▼  (display thread reads latest result)
//   [Display/Main Thread] ──→ LivePipeline::getLatest()
//
// Key design decisions:
//
//   SINGLE-SLOT RESULT BUFFER (not a queue):
//     The display thread always wants the LATEST result, not an old one.
//     A queue would introduce latency proportional to its length.
//     A single slot (swap on write, copy on read) gives minimum latency.
//
//   FRAME-DROP STRATEGY:
//     The processing thread grabs the latest frame and immediately starts
//     processing. If a new camera frame arrives during processing (likely,
//     since SGBM takes 100-300ms and cameras run at ~10 FPS), the next
//     iteration picks up the newer frame. Old frames are silently discarded.
//     This is the correct strategy for real-time display.
//
//   FRAME VALIDATION:
//     Before processing, the pipeline checks:
//     1. Both frames are non-empty.
//     2. Both frames are fresh (age < maxFrameAgeMs).
//     3. Timestamps are synchronized (|t_L - t_R| < maxSyncDiffMs).
//     Stale or unsynced pairs are skipped.
//
//   SGBM PARAM UPDATES:
//     setParams() signals the processing thread via paramsChanged_ flag.
//     The thread applies new params at the START of the next iteration,
//     not mid-computation. No mutex needed for params (atomic flag + copy).
// ─────────────────────────────────────────────────────────────────────────────
class LivePipeline {
public:

    // Configuration
    struct Config {
        double maxFrameAgeMs  = 500.0;  // reject frames older than this
        double maxSyncDiffMs  = 100.0;  // reject pairs with > this timestamp gap
        bool   showRectified  = false;  // show rectified frames alongside disparity
        bool   showDepth      = true;   // compute and show depth map
    };

    // Per-frame result — everything the display thread needs
    struct Result {
        cv::Mat disparity;      // raw 16S disparity from SGBM
        cv::Mat dispVis;        // 8-bit false-color disparity
        cv::Mat depthM;         // CV_32F depth in meters (or empty)
        cv::Mat depthVis;       // 8-bit false-color depth (or empty)
        cv::Mat rectLeft;       // rectified left frame (if showRectified)
        cv::Mat rectRight;      // rectified right frame (if showRectified)

        double  processingMs  = 0.0;  // time for one full rectify+SGBM+depth cycle
        int     frameIndex    = 0;
        int     droppedFrames = 0;    // frames skipped due to age/sync issues
    };

    // ─────────────────────────────────────────────────────────────────────────
    LivePipeline(Rectifier& rectifier, SGBMProcessor& sgbm, const Config& cfg);
    ~LivePipeline();

    // Start the processing thread. Cameras must already be streaming.
    void start(CameraStream& camL, CameraStream& camR);

    // Signal stop and join the processing thread. Blocks until done.
    void stop();

    // Thread-safe: copy the latest result. Returns false if no result yet.
    bool getLatest(Result& out) const;

    // Thread-safe: update SGBM parameters. Applied at next frame boundary.
    void setParams(const SGBMProcessor::Params& p);

    bool isRunning() const { return running_.load(); }

    // Runtime metrics
    struct Metrics {
        double avgProcessingMs = 0.0;
        double actualFPS       = 0.0;
        int    totalFrames     = 0;
        int    droppedFrames   = 0;
    };
    Metrics getMetrics() const;

private:
    Rectifier&     rectifier_;
    SGBMProcessor& sgbm_;
    Config         cfg_;
    CameraStream*  camL_ = nullptr;
    CameraStream*  camR_ = nullptr;

    std::thread       thread_;
    std::atomic<bool> running_{false};

    // Single-slot result buffer
    mutable std::mutex resultMtx_;
    Result             latestResult_;
    bool               hasResult_ = false;

    // Parameter update signaling
    std::mutex              paramsMtx_;
    SGBMProcessor::Params   pendingParams_;
    std::atomic<bool>       paramsChanged_{false};

    // Metrics (approximate — no mutex, slight data races OK for display)
    mutable Metrics metrics_;

    void loop();  // processing thread body

    // Returns true if frames are fresh and synchronized
    bool framesValid(const cv::Mat& fL, const cv::Mat& fR,
                     std::chrono::steady_clock::time_point tsL,
                     std::chrono::steady_clock::time_point tsR) const;
};
