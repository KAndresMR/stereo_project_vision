#include "LivePipeline.hpp"
#include "DiagnosticLogger.hpp"
#include <numeric>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::milliseconds;
using Fms   = std::chrono::duration<double, std::milli>;

// ─────────────────────────────────────────────────────────────────────────────
LivePipeline::LivePipeline(Rectifier& rectifier, SGBMProcessor& sgbm,
                            const Config& cfg)
    : rectifier_(rectifier), sgbm_(sgbm), cfg_(cfg) {}

LivePipeline::~LivePipeline() { stop(); }

// ─────────────────────────────────────────────────────────────────────────────
void LivePipeline::start(CameraStream& camL, CameraStream& camR) {
    if (running_) return;
    camL_    = &camL;
    camR_    = &camR;
    running_ = true;
    thread_  = std::thread(&LivePipeline::loop, this);
    Log::info("LivePipeline", "Processing thread started");
}

void LivePipeline::stop() {
    if (!running_) return;
    running_ = false;
    if (thread_.joinable()) thread_.join();
    Log::info("LivePipeline", "Processing thread stopped");
}

// ─────────────────────────────────────────────────────────────────────────────
bool LivePipeline::framesValid(const cv::Mat& fL, const cv::Mat& fR,
                                Clock::time_point tsL,
                                Clock::time_point tsR) const {
    if (fL.empty() || fR.empty()) return false;

    auto now = Clock::now();
    double ageL = Fms(now - tsL).count();
    double ageR = Fms(now - tsR).count();

    if (ageL > cfg_.maxFrameAgeMs || ageR > cfg_.maxFrameAgeMs) {
        metrics_.droppedFrames++;
        return false;
    }

    double syncDiff = std::abs(Fms(tsL - tsR).count());
    if (syncDiff > cfg_.maxSyncDiffMs) {
        metrics_.droppedFrames++;
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// loop — processing thread body
//
// Runs continuously while running_ == true.
// Does NOT sleep between iterations — if SGBM is slower than cameras,
// it naturally processes at camera FPS. If faster, it re-processes the
// same frame (slightly wasteful but keeps latency minimal).
//
// Consider adding a condition_variable if CPU usage is a concern:
//   wait until camera timestamp changes before processing next frame.
// ─────────────────────────────────────────────────────────────────────────────
void LivePipeline::loop() {
    Log::info("LivePipeline", "Thread running");

    // For FPS measurement
    auto fpsTimer   = Clock::now();
    int  frameCount = 0;

    // Rolling average for processingMs
    std::deque<double> processTimes;

    while (running_) {

        // ── Apply pending param update ─────────────────────────────────────
        if (paramsChanged_.exchange(false)) {
            std::lock_guard<std::mutex> l(paramsMtx_);
            sgbm_.setParams(pendingParams_);
        }

        // ── Grab latest synchronized frame pair ───────────────────────────
        cv::Mat rawL, rawR;
        Clock::time_point tsL, tsR;
        {
            std::lock_guard<std::mutex> lL(camL_->frame_mtx);
            std::lock_guard<std::mutex> lR(camR_->frame_mtx);
            rawL = camL_->frame.clone();
            rawR = camR_->frame.clone();
            tsL  = camL_->timestamp;
            tsR  = camR_->timestamp;
        }

        if (!framesValid(rawL, rawR, tsL, tsR)) {
            std::this_thread::sleep_for(Ms(5));
            continue;
        }

        // ── Time the full processing cycle ────────────────────────────────
        auto t0 = Clock::now();

        // ── Rectify ───────────────────────────────────────────────────────
        auto [rectL, rectR] = rectifier_.rectify(rawL, rawR);
        if (rectL.empty() || rectR.empty()) continue;

        // ── SGBM ──────────────────────────────────────────────────────────
        cv::Mat rawDisp  = sgbm_.compute(rectL, rectR);
        cv::Mat filtDisp = sgbm_.postprocess(rawDisp, rectL);
        cv::Mat smoothDisp = sgbm_.temporalSmooth(filtDisp);
        cv::Mat dispVis  = sgbm_.visualize(smoothDisp);

        // ── Depth ─────────────────────────────────────────────────────────
        cv::Mat depthM, depthVis;
        if (cfg_.showDepth) {
            depthM   = sgbm_.toDepth(smoothDisp, rectifier_.maps().Q);
            depthVis = sgbm_.visualizeDepth(depthM);
        }

        double ms = Fms(Clock::now() - t0).count();

        // ── Publish result ────────────────────────────────────────────────
        {
            std::lock_guard<std::mutex> l(resultMtx_);
            latestResult_.disparity    = smoothDisp;
            latestResult_.dispVis      = dispVis;
            latestResult_.depthM       = depthM;
            latestResult_.depthVis     = depthVis;
            latestResult_.processingMs = ms;
            latestResult_.frameIndex++;
            latestResult_.droppedFrames = metrics_.droppedFrames;

            if (cfg_.showRectified) {
                latestResult_.rectLeft  = rectL;
                latestResult_.rectRight = rectR;
            }
            hasResult_ = true;
        }

        // ── Update metrics ────────────────────────────────────────────────
        processTimes.push_back(ms);
        if (processTimes.size() > 30) processTimes.pop_front();
        metrics_.avgProcessingMs = std::accumulate(
            processTimes.begin(), processTimes.end(), 0.0) / processTimes.size();

        frameCount++;
        metrics_.totalFrames++;
        double elapsed = Fms(Clock::now() - fpsTimer).count() / 1000.0;
        if (elapsed >= 2.0) {
            metrics_.actualFPS = frameCount / elapsed;
            Log::info("LivePipeline",
                "FPS=" + std::to_string((int)metrics_.actualFPS) +
                " proc=" + std::to_string((int)metrics_.avgProcessingMs) + "ms" +
                " dropped=" + std::to_string(metrics_.droppedFrames));
            frameCount = 0;
            fpsTimer   = Clock::now();
        }
    }

    Log::info("LivePipeline", "Thread exiting");
}

// ─────────────────────────────────────────────────────────────────────────────
bool LivePipeline::getLatest(Result& out) const {
    std::lock_guard<std::mutex> l(resultMtx_);
    if (!hasResult_) return false;
    out = latestResult_;  // full copy — caller owns the data
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
void LivePipeline::setParams(const SGBMProcessor::Params& p) {
    {
        std::lock_guard<std::mutex> l(paramsMtx_);
        pendingParams_ = p;
    }
    paramsChanged_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
LivePipeline::Metrics LivePipeline::getMetrics() const {
    return metrics_;
}
