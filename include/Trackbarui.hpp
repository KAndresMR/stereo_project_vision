#pragma once
#include <opencv2/opencv.hpp>
#include <mutex>
#include <atomic>
#include <string>
#include "SGBMProcessor.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// TrackbarUI
//
// Encapsula toda la lógica de barras de desplazamiento de OpenCV para ajuste
// de parámetros SGBM.
//
// Decisiones de diseño:
//
//   1. SINGLETON mediante puntero estático.
//      Los callbacks de OpenCV deben ser funciones C puras o lambdas sin captura.
//      La única forma limpia de acceder al estado de clase desde ellos es
//      un puntero estático. Una instancia TrackbarUI por proceso — aceptable aquí.
//
//   2. El mapeo de INT→PARAM ocurre al leer, no al escribir.
//      Los sliders guardan valores enteros puros (requerimiento de OpenCV).
//      La conversión a rangos SGBM correctos (blockSize debe ser impar, 
//      numDisparities % 16 == 0) pasa en getParams(), no en el callback.
//      Esto simplifica los callbacks a una sola operación de escritura.
//
//   3. Hilo seguro vía mutex + flag atómico.
//      El hilo de procesamiento llama a getParams() y hasChanged() concurrentemente
//      con el hilo de display que maneja los eventos de OpenCV. El mutex protege
//      los valores del slider; el flag atómico evita el mutex en el bucle caliente.
//
//   4. SGBM no se recrea en cada arrastre del slider.
//      hasChanged() retorna true una vez y luego se resetea. Quien llama recrea
//      SGBM solo cuando el flag está activo, no cada frame.
//
// Uso:
//   TrackbarUI ui;
//   ui.create("Controles SGBM", initialParams);
//
//   while (running) {
//       cv::waitKey(1);                   // procesa eventos de la ventana
//       if (ui.hasChanged()) {
//           processor.setParams(ui.getParams());
//       }
//   }
// ─────────────────────────────────────────────────────────────────────────────
class TrackbarUI {
public:

    explicit TrackbarUI() = default;

    // Crea la ventana con nombre y todas las barras de control (trackbars).
    // Debe llamarse desde el hilo principal/de interfaz gráfica.
    void create(const std::string& windowName,
                const SGBMProcessor::Params& initial) {
        windowName_ = windowName;
        cv::namedWindow(windowName_, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName_, 400, 350);

        // Guardar valores iniciales de los controles a partir de los parámetros
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

        // ── Parámetros SGBM ──────────────────────────────────────────────────
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

        // ── Parámetros WLS ────────────────────────────────────────────────────
        cv::createTrackbar("WLS lambda (x100)", windowName_,
            &sliders_.wlsLambda, 200, cbAny, nullptr);

        cv::createTrackbar("WLS sigma (x10)", windowName_,
            &sliders_.wlsSigma, 30, cbAny, nullptr);

        changed_ = true;  // forzar aplicación inicial
    }

    // Retorna true una vez por evento de cambio, luego se reinicia.
    bool hasChanged() {
        return changed_.exchange(false);
    }

    // Thread-safe: convierte valores puros de la UI a parámetros SGBM válidos.
    SGBMProcessor::Params getParams() const {
        std::lock_guard<std::mutex> l(mtx_);
        SGBMProcessor::Params p;

        // blockSize: control n → 2n+1 (asegura impar, min 3)
        p.blockSize = std::max(3, sliders_.blockSize * 2 + 1);

        // numDisparities: control n → 16n (múltiplo de 16, min 16)
        p.numDisparities = std::max(16, sliders_.numDisparities * 16);

        p.uniquenessRatio  = std::max(0, sliders_.uniquenessRatio);
        p.speckleWindowSize = sliders_.speckleWinSize;
        p.speckleRange     = sliders_.speckleRange;
        p.disp12MaxDiff    = sliders_.disp12MaxDiff;
        p.preFilterCap     = std::max(1, sliders_.preFilterCap);

        // WLS: control n → λ = 100n, σ = n/10
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

    // Valores int para OpenCV (sliders deben ser int*)
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

    // Puntero singleton para callback estilo C
    static TrackbarUI* instance_;

    // Único callback para todos los controles — solo marca el flag.
    // El valor real está en el int* pasado por OpenCV (ya actualizado).
    static void cbAny(int, void*) {
        if (instance_) instance_->changed_ = true;
    }
};

// Definición del miembro estático (se pone en UN .cpp que incluya este header)
inline TrackbarUI* TrackbarUI::instance_ = nullptr;