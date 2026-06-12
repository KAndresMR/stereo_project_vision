#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "CalibrationConfig.hpp"
#include "ChessboardDetector.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// StereoCalibrator — Fase 2 del pipeline
//
// Responsabilidad:
//   Dado un dataset de imágenes estéreo y las propiedades intrínsecas individuales 
//   precalculadas (desde MonoCalibrator), calcular la relación geométrica entre
//   las dos cámaras: rotación R, traslación T, matriz esencial E, y matriz
//   fundamental F.
//
// Lo que NO hace:
//   - calibración de cámara individual       (MonoCalibrator)
//   - rectificación                          (Rectifier)
//   - disparidad o profundidad               (SGBMProcessor)
//
// Decisión clave de diseño: CALIB_FIX_INTRINSIC
//   Fijamos K_left, dist_left, K_right, dist_right a sus valores precalculados
//   y solo optimizamos R y T. Este es el enfoque profesional correcto —
//   produce una geometría más estable con datasets más pequeños.
//
// Uso:
//   StereoCalibrator sc(config);
//   auto result = sc.calibrate();
//   if (result.success) sc.saveYAML(result); 
// ─────────────────────────────────────────────────────────────────────────────
class StereoCalibrator {
public:

    // ── Resultado ─────────────────────────────────────────────────────────────
    struct Result {
        bool success = false;

        cv::Mat R;           // Rotacion 3×3:   camara derecha relativa a la izquierda
        cv::Mat T;           // Traslacion 3×1: origen camara derecha en coords de la izq (metros)
        cv::Mat E;           // Matriz esencial 3×3
        cv::Mat F;           // Matriz fundamental 3×3

        double rpe        = 0.0;  // Error de reproyeccion RMS estereo
        double baselineM  = 0.0;  // |T| en metros (≈ separacion fisica)
        int    pairsUsed  = 0;
        int    pairsTotal = 0;
        cv::Size imageSize;
    };

    explicit StereoCalibrator(const CalibrationConfig& config);

    // Pipeline completo offline: carga intrínsecas, detecta pares, ejecuta stereoCalibrate.
    Result calibrate() const;

    // Guarda R, T, E, F, Q, R1, R2, P1, P2 en stereo.yaml.
    // También ejecuta internamente stereoRectify para calcular y guardar la matriz Q,
    // de manera que todo lo necesario para la siguiente fase esté en un solo archivo.
    bool saveYAML(const Result& result) const;

    void printSummary(const Result& result) const;

private:
    CalibrationConfig  config_;
    ChessboardDetector detector_;

    // Carga K y distCoeffs desde left.yaml / right.yaml.
    bool loadIntrinsics(cv::Mat& K_left,  cv::Mat& dist_left,
                        cv::Mat& K_right, cv::Mat& dist_right,
                        cv::Size& imageSize) const;

    // Detecta esquinas simultáneamente en un par estéreo.
    // Retorna true solo si AMBAS imágenes producen un conjunto completo de esquinas.
    bool detectPair(const std::string& leftPath,
                    const std::string& rightPath,
                    std::vector<cv::Point2f>& cornersL,
                    std::vector<cv::Point2f>& cornersR,
                    cv::Size& imageSize) const;

    std::vector<cv::Point3f> buildObjectPoints() const;

    void validateResult(const Result& result) const;
};