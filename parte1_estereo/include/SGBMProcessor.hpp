#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <deque>
#include <mutex>

// ─────────────────────────────────────────────────────────────────────────────
// SGBMProcessor — Emparejamiento Estéreo + WLS + Suavizado Temporal
//
// Modelo de propiedad:
//   - Posee el StereoSGBM (matcher izquierdo) y el matcher derecho.
//   - Posee el filtro WLS.
//   - Posee el buffer de suavizado temporal.
//   - Sin estado respecto a los frames — compute() es pura.
//     Excepción: temporalSmooth() tiene estado (buffer histórico).
//
// Seguridad de hilos:
//   - setParams() NO es seguro entre hilos — llamar solo desde el hilo de display
//     cuando el hilo de procesamiento está pausado.
//   - compute(), postprocess(), normalize() SÍ son seguros de llamar desde un
//     hilo de procesamiento concurrentemente con el hilo de display.
//
// Nota sobre el filtro WLS:
//   WLS requiere una "disparidad derecha" calculada ejecutando SGBM en dirección
//   inversa (derecha→izquierda). Esto cuesta ~2× CPU por frame pero mejora
//   significativamente los bordes y rellena las regiones de oclusión.
//   Controlado por el flag useWLS en Params.
// ─────────────────────────────────────────────────────────────────────────────
class SGBMProcessor {
public:

    struct Params {
        // ── SGBM ─────────────────────────────────────────────────────────────
        int minDisparity     = 16;
        int numDisparities   = 96;   // debe ser múltiplo de 16
        int blockSize        = 9;    // debe ser impar

        // Penalizaciones de suavidad. 0 → se calculan automáticamente desde blockSize.
        int P1               = 0;
        int P2               = 0;

        int disp12MaxDiff    = 1;
        int preFilterCap     = 63;  // valor por defecto: 31
        int uniquenessRatio  = 20;
        int speckleWindowSize = 250;
        int speckleRange     = 2;
        int mode             = cv::StereoSGBM::MODE_SGBM_3WAY;

        // ── Filtro WLS ───────────────────────────────────────────────────────
        bool   useWLS    = true;
        double wlsLambda = 10000.0;
        double wlsSigma  = 0.6;

        // ── Suavizado temporal ────────────────────────────────────────────────
        bool  useTemporalSmoothing = true;
        int   temporalWindow       = 4;    // frames a mezclar
        float temporalAlpha        = 0.4f; // peso EMA [0=congelado, 1=sin suavizado]

        // ── Salida de profundidad ─────────────────────────────────────────────
        float minDepthM = 0.10f;
        float maxDepthM = 5.00f;

        // ── Preprocesamiento CLAHE ──────────────────────────────────────────
        double claheClipLimit = 2.5;
        int claheTileSize = 8;

        // ── Preprocesamiento Gaussian Blur ──────────────────────────────────────────
        double preBlurSigma = 0.8;
        
    };

    explicit SGBMProcessor(const Params& params);

    // Reinicia todos los matchers y el filtro WLS. Llamar tras setParams().
    void setParams(const Params& p);
    const Params& getParams() const { return params_; }

    // ── Pasos del pipeline principal ──────────────────────────────────────────

    // Paso 1: Calcular disparidad cruda (CV_16S, disp_real = valor / 16.0).
    // rectLeft y rectRight deben estar ya sin distorsión y rectificadas.
    // Aplica internamente CLAHE antes de SGBM para mejor contraste.
    cv::Mat compute(const cv::Mat& rectLeft, const cv::Mat& rectRight);

    // Paso 2: Post-filtro WLS (rellena huecos, afila bordes).
    // Pasar el resultado de compute() y rectLeft como guía de bordes.
    // Si useWLS==false, retorna rawDisparity sin cambios.
    cv::Mat postprocess(const cv::Mat& rawDisparity,
                        const cv::Mat& rectLeft);

    // Paso 3: Suavizado temporal EMA entre frames.
    // Reduce el parpadeo frame a frame a costa de un leve retraso.
    // Si useTemporalSmoothing==false, retorna la entrada sin cambios.
    cv::Mat temporalSmooth(const cv::Mat& filteredDisparity);

    // Paso 4: Convertir disparidad 16S a falso color de 8 bits para mostrar.
    // COLORMAP_TURBO: azul=lejos, rojo=cerca.
    cv::Mat visualize(const cv::Mat& disparity) const;

    // ── Conversión a profundidad ──────────────────────────────────────────────

    // Convierte disparidad filtrada → profundidad métrica (Z en metros) usando la matriz Q.
    // Los píxeles inválidos (disp <= 0) y profundidades fuera de rango se ponen en 0.
    // Retorna un mapa de profundidad CV_32F.
    cv::Mat toDepth(const cv::Mat& disparity, const cv::Mat& Q) const;

    // Colorizar el mapa de profundidad para visualización (cerca=rojo, lejos=azul).
    cv::Mat visualizeDepth(const cv::Mat& depthM) const;

    // Getter para obtener la imagen procesada con CLAHE y mostrarla en el dashboard
    cv::Mat getEnhancedLeft() const { return enhancedL_; }

private:
    Params params_;

    cv::Ptr<cv::StereoSGBM>                    leftMatcher_;
    cv::Ptr<cv::StereoMatcher>                  rightMatcher_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter>   wlsFilter_;
    cv::Ptr<cv::CLAHE>                          clahe_;

    // Estado del suavizado temporal (promedio acumulado CV_32F)
    cv::Mat  smoothed_;
    bool     smoothedValid_ = false;

    // Disparidad derecha (necesaria para WLS, calculada en compute() cuando useWLS=true)
    cv::Mat  rightDisp_;

    void init();  // (re)crear todos los matchers desde params_
    
    // Variable para guardar el resultado del CLAHE
    cv::Mat enhancedL_;
};