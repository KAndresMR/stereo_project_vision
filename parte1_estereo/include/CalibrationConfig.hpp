#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// CalibrationConfig — fuente única de verdad para todos los parámetros del pipeline.
//
// Organizado por fase para saber exactamente qué valores afectan cada paso.
// ─────────────────────────────────────────────────────────────────────────────
struct CalibrationConfig {

    // ── Tablero de ajedrez ───────────────────────────────────────────────────
    cv::Size boardSize{9, 6};      // esquinas internas (no cuadros)
    float    squareSizeM = 0.025f; // tamaño físico del cuadro en METROS (25mm)

    // ── Sesión ───────────────────────────────────────────────────────────────
    int         targetPairs = 30;
    std::string datasetDir  = "calib_pairs";   // raíz para imágenes left/ y right/

    // ── Archivos YAML de salida ───────────────────────────────────────────────
    // Fase 1 — calibración individual
    std::string leftYaml  = "calib_pairs/calib/left.yaml";
    std::string rightYaml = "calib_pairs/calib/right.yaml";
    // Fase 2 — calibración estéreo
    std::string stereoYaml = "calib_pairs/calib/stereo.yaml";

    // ── Flags de detección ───────────────────────────────────────────────────
    int findFlags = cv::CALIB_CB_ADAPTIVE_THRESH
                  | cv::CALIB_CB_NORMALIZE_IMAGE
                  | cv::CALIB_CB_FAST_CHECK;

    // ── Refinamiento sub-píxel ───────────────────────────────────────────────
    cv::TermCriteria subPixCriteria{
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
        30, 0.001
    };
    cv::Size subPixWinSize{11, 11};

    // ── CLAHE (compensación de iluminación) ──────────────────────────────────
    // Cuando el brillo medio del frame cae por debajo de brightnessThreshold,
    // se aplica CLAHE antes de la detección de esquinas para recuperar el contraste.
    // Se puede dejar activado siempre — CLAHE no hace nada en frames bien iluminados.
    bool   autoEnhance         = true;
    float  brightnessThreshold = 80.0f;  // valor medio del píxel: 0–255
    double claheClipLimit      = 2.0;    // mayor valor = más agresivo
    cv::Size claheTileSize{8, 8};

    // ── Captura ───────────────────────────────────────────────────────────────
    int cooldownMs = 800;  // ms mínimos entre capturas (previene doble disparo)

    // ── Flags de calibración estéreo ─────────────────────────────────────────
    // FIX_INTRINSIC: usa K/distorsión de la Fase 1 tal cual, solo optimiza R/T.
    // Es el enfoque correcto cuando la calibración individual fue buena.
    int stereoFlags = cv::CALIB_FIX_INTRINSIC;
};