#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "CalibrationConfig.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Rectifier — Fase 3 del pipeline
//
// Responsabilidad:
//   Dada la calibración estéreo (R, T, K, dist para cada cámara),
//   calcular el remapeo a nivel de píxel que transforma las imágenes crudas
//   de la cámara en imágenes rectificadas donde:
//     - las líneas epipolares son perfectamente horizontales
//     - los puntos correspondientes en la imagen izquierda y derecha yacen en la misma fila
//     - la correspondencia estéreo (SGBM) solo necesita escanear una fila por píxel
//
// Dos modos de uso:
//   OFFLINE: rectificar imágenes guardadas del dataset (para depuración/verificación)
//   EN VIVO: rectificar frames de los streams del ESP32 en tiempo real
//
// Uso:
//   Rectifier rect(config);
//   if (rect.compute()) {
//       auto [left_r, right_r] = rect.rectify(rawLeft, rawRight);
//       cv::Mat debug = rect.drawEpipolarLines(left_r, right_r);
//   }
// ─────────────────────────────────────────────────────────────────────────────
class Rectifier {
public:

    // Todos los datos de rectificación precalculados necesarios en tiempo de ejecución.
    struct Maps {
        cv::Mat map1x, map1y;   // mapeo de píxeles para cámara IZQUIERDA
        cv::Mat map2x, map2y;   // mapeo de píxeles para cámara DERECHA
        cv::Mat R1, R2;         // rotación de rectificación para cada cámara
        cv::Mat P1, P2;         // matrices de proyección tras la rectificación
        cv::Mat Q;              // matriz 4×4 para convertir disparidad a profundidad
        cv::Size imageSize;
        bool ready = false;
    };

    explicit Rectifier(const CalibrationConfig& config);

    // Carga todos los YAMLs y calcula las tablas de mapeo.
    // Debe llamarse antes de rectify() o drawEpipolarLines().
    bool compute();

    // Aplica rectificación a un par estéreo.
    // Entrada: frames crudos de las cámaras (o del disco)
    // Salida:  par de imágenes rectificadas (izquierda, derecha)
    std::pair<cv::Mat, cv::Mat> rectify(const cv::Mat& rawLeft,
                                         const cv::Mat& rawRight) const;

    // Visualización de depuración: par rectificado lado a lado con líneas verdes
    // horizontales superpuestas. Si la rectificación es correcta, el mismo punto físico
    // aparece en la MISMA línea en ambas imágenes.
    cv::Mat drawEpipolarLines(const cv::Mat& rectLeft,
                               const cv::Mat& rectRight,
                               int lineSpacing = 40) const;

    // Ejecuta una verificación visual offline en los pares guardados del dataset.
    // Carga cada par, rectifica, dibuja líneas epipolares, y muestra una ventana.
    // Presionar cualquier tecla para avanzar, ESC para salir.
    void previewDataset() const;

    bool isReady() const { return maps_.ready; }
    const Maps& maps() const { return maps_; }

private:
    CalibrationConfig config_;
    Maps maps_;

    bool loadYAMLs(cv::Mat& K_left,  cv::Mat& dist_left,
                   cv::Mat& K_right, cv::Mat& dist_right,
                   cv::Mat& R,       cv::Mat& T,
                   cv::Mat& R1,      cv::Mat& R2,
                   cv::Mat& P1,      cv::Mat& P2,
                   cv::Mat& Q,
                   cv::Size& imageSize) const;
};