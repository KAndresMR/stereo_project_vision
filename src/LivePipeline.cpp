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
    Log::info("LivePipeline", "Hilo de procesamiento iniciado");
}

void LivePipeline::stop() {
    if (!running_) return;
    running_ = false;
    if (thread_.joinable()) thread_.join();
    Log::info("LivePipeline", "Hilo de procesamiento detenido");
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
// loop — cuerpo del hilo de procesamiento
//
// Corre continuamente mientras running_ == true.
// NO duerme entre iteraciones — si SGBM es más lento que las cámaras,
// naturalmente procesa a los FPS de las cámaras. Si es más rápido, reprocesa
// el mismo frame (un poco de desperdicio pero mantiene latencia mínima).
//
// Considerar añadir un condition_variable si el uso del CPU es un problema:
//   esperar hasta que la marca de tiempo de la cámara cambie antes del próximo frame.
// ─────────────────────────────────────────────────────────────────────────────
void LivePipeline::loop() {
    Log::info("LivePipeline", "Hilo corriendo");

    // Para la medición de FPS
    auto fpsTimer   = Clock::now();
    int  frameCount = 0;

    // Promedio móvil para processingMs
    std::deque<double> processTimes;

    while (running_) {

        // ── Aplicar actualización de parámetros pendiente ──────────────────
        if (paramsChanged_.exchange(false)) {
            std::lock_guard<std::mutex> l(paramsMtx_);
            sgbm_.setParams(pendingParams_);
        }

        // ── Tomar el último par de frames sincronizados ─────────────────────
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

        // ── Medir el tiempo del ciclo de procesamiento completo ──────────────
        auto t0 = Clock::now();

        // ── Rectificar ────────────────────────────────────────────────────────
        auto [rectL, rectR] = rectifier_.rectify(rawL, rawR);
        if (rectL.empty() || rectR.empty()) continue;

        // ── SGBM ─────────────────────────────────────────────────────────────
        cv::Mat rawDisp  = sgbm_.compute(rectL, rectR);
        cv::Mat filtDisp = sgbm_.postprocess(rawDisp, rectL);
        cv::Mat smoothDisp = sgbm_.temporalSmooth(filtDisp);
        cv::Mat dispVis  = sgbm_.visualize(smoothDisp);

        // ── Profundidad ───────────────────────────────────────────────────────
        cv::Mat depthM, depthVis;
        if (cfg_.showDepth) {
            depthM   = sgbm_.toDepth(smoothDisp, rectifier_.maps().Q);
            depthVis = sgbm_.visualizeDepth(depthM);
        }

        double ms = Fms(Clock::now() - t0).count();

        // ── Publicar resultado ────────────────────────────────────────────────
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

        // ── Actualizar métricas ───────────────────────────────────────────────
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
                " descarts=" + std::to_string(metrics_.droppedFrames));
            frameCount = 0;
            fpsTimer   = Clock::now();
        }
    }

    Log::info("LivePipeline", "Hilo finalizando");
}

// ─────────────────────────────────────────────────────────────────────────────
bool LivePipeline::getLatest(Result& out) const {
    std::lock_guard<std::mutex> l(resultMtx_);
    if (!hasResult_) return false;
    out = latestResult_;  // copia completa — quien llama posee los datos
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
