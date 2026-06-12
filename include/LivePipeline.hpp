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
// LivePipeline — Hilo de procesamiento estéreo en tiempo real
//
// Modelo de hilos:
//
//   [Hilo Cámara L] ──→ CameraStream::frame (protegido por mutex)
//   [Hilo Cámara R] ──→ CameraStream::frame (protegido por mutex)
//                                   │
//                                   ▼  (hilo de proc. toma el ÚLTIMO par)
//   [Hilo Proc.] ──→ rectificación → SGBM → profundidad → Result (un solo slot)
//                                   │
//                                   ▼  (hilo de UI lee el último resultado)
//   [Hilo Principal/UI] ──→ LivePipeline::getLatest()
//
// Decisiones clave de diseño:
//
//   BUFFER DE RESULTADO DE UN SOLO SLOT (no es una cola):
//     El hilo de visualización siempre quiere el ÚLTIMO resultado, no uno viejo.
//     Una cola introduciría latencia proporcional a su tamaño.
//     Un solo slot (intercambio al escribir, copia al leer) da latencia mínima.
//
//   ESTRATEGIA DE DESCARTE DE FRAMES:
//     El hilo de procesamiento toma el último frame e inmediatamente empieza
//     a procesar. Si llega un nuevo frame durante el proceso (probable,
//     ya que SGBM toma 100-300ms y las cámaras van a ~10 FPS), la siguiente
//     iteración toma el frame más nuevo. Los frames viejos se descartan
//     silenciosamente. Esta es la estrategia correcta para vista en tiempo real.
//
//   VALIDACIÓN DE FRAMES:
//     Antes de procesar, el pipeline verifica:
//     1. Ambos frames no están vacíos.
//     2. Ambos frames son frescos (edad < maxFrameAgeMs).
//     3. Las marcas de tiempo están sincronizadas (|t_L - t_R| < maxSyncDiffMs).
//     Pares viejos o desincronizados son omitidos.
//
//   ACTUALIZACIONES DE PARÁMETROS SGBM:
//     setParams() le avisa al hilo de procesamiento vía el flag paramsChanged_.
//     El hilo aplica los nuevos parámetros al INICIO de la siguiente iteración,
//     no en medio de cálculos. No se necesita mutex para parámetros 
//     (flag atómico + copia).
// ─────────────────────────────────────────────────────────────────────────────
class LivePipeline {
public:

    // Configuración
    struct Config {
        double maxFrameAgeMs  = 500.0;  // rechazar frames más viejos que esto
        double maxSyncDiffMs  = 100.0;  // rechazar pares con > esta brecha de tiempo
        bool   showRectified  = false;  // mostrar frames rectificados junto a disparidad
        bool   showDepth      = true;   // calcular y mostrar mapa de profundidad
    };

    // Resultado por frame — todo lo que necesita el hilo de visualización
    struct Result {
        cv::Mat disparity;      // disparidad 16S cruda de SGBM
        cv::Mat dispVis;        // disparidad en falso color (8-bit)
        cv::Mat depthM;         // profundidad CV_32F en metros (o vacío)
        cv::Mat depthVis;       // profundidad en falso color (8-bit, o vacío)
        cv::Mat rectLeft;       // frame izquierdo rectificado (si showRectified)
        cv::Mat rectRight;      // frame derecho rectificado (si showRectified)

        double  processingMs  = 0.0;  // tiempo de un ciclo rectificar+SGBM+profundidad
        int     frameIndex    = 0;
        int     droppedFrames = 0;    // frames saltados por problemas de edad/sync
    };

    // ─────────────────────────────────────────────────────────────────────────
    LivePipeline(Rectifier& rectifier, SGBMProcessor& sgbm, const Config& cfg);
    ~LivePipeline();

    // Iniciar el hilo de procesamiento. Las cámaras deben estar ya transmitiendo.
    void start(CameraStream& camL, CameraStream& camR);

    // Señalar parada y hacer join del hilo de procesamiento. Bloquea hasta finalizar.
    void stop();

    // Thread-safe: copia el último resultado. Retorna false si no hay resultado aún.
    bool getLatest(Result& out) const;

    // Thread-safe: actualiza parámetros SGBM. Aplicado en el siguiente frame.
    void setParams(const SGBMProcessor::Params& p);

    bool isRunning() const { return running_.load(); }

    // Métricas en tiempo de ejecución
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

    // Buffer de resultado de un solo slot
    mutable std::mutex resultMtx_;
    Result             latestResult_;
    bool               hasResult_ = false;

    // Señalización de actualización de parámetros
    std::mutex              paramsMtx_;
    SGBMProcessor::Params   pendingParams_;
    std::atomic<bool>       paramsChanged_{false};

    // Métricas (aproximadas — sin mutex, leves desincronizaciones están bien para UI)
    mutable Metrics metrics_;

    void loop();  // cuerpo del hilo de procesamiento

    // Retorna true si los frames están frescos y sincronizados
    bool framesValid(const cv::Mat& fL, const cv::Mat& fR,
                     std::chrono::steady_clock::time_point tsL,
                     std::chrono::steady_clock::time_point tsR) const;
};
