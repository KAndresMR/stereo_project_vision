#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "CalibrationConfig.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// DetectionResult — todo lo que el detector sabe sobre un intento de detección.
// ─────────────────────────────────────────────────────────────────────────────
struct DetectionResult {
    bool   found       = false;
    bool   claheUsed   = false;   // ¿se aplicó CLAHE antes de la detección?
    double brightness  = 0.0;     // valor medio del píxel del frame en escala de grises (0–255)
    double contrast    = 0.0;     // contraste de Michelson del frame (0–1)
    int    cornersFound    = 0;   // cuántas esquinas se encontraron (aunque no estén completas)
    int    cornersExpected = 0;   // boardSize.width * boardSize.height

    std::vector<cv::Point2f> corners;  // esquinas refinadas (válidas solo cuando found==true)
};

// ─────────────────────────────────────────────────────────────────────────────
// ChessboardDetector
//
// Pipeline por cada llamada:
//   1. Medir brillo — decidir si se necesita CLAHE
//   2. Aplicar CLAHE opcionalmente a una copia de trabajo del frame en grises
//   3. findChessboardCorners (FAST_CHECK rechaza rápido cuando no hay tablero)
//   4. cornerSubPix — refinamiento sub-píxel (solo cuando se encuentra)
//   5. drawChessboardCorners en el frame de color para visualización (solo cuando se encuentra)
// ─────────────────────────────────────────────────────────────────────────────
class ChessboardDetector {
public:
    explicit ChessboardDetector(const CalibrationConfig& config);

    // grayFrame    — CV_8UC1 en escala de grises (no se modifica)
    // displayFrame — CV_8UC3 frame de color; se dibujan las esquinas cuando se encuentran
    DetectionResult detect(const cv::Mat& grayFrame,
                           cv::Mat&       displayFrame) const;

private:
    CalibrationConfig config_;

    double measureBrightness(const cv::Mat& gray) const;
    double measureContrast(const cv::Mat& gray) const;
    cv::Mat applyClahe(const cv::Mat& gray) const;
};