#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "CalibrationConfig.hpp"
#include "ChessboardDetector.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// MonoCalibrator — Fase 1 del pipeline estéreo
//
// Responsabilidad: dada una carpeta de imágenes del tablero de ajedrez
// de UNA sola cámara, calcular la matriz intrínseca K y los coeficientes de
// distorsión, y luego guardarlos en un archivo YAML.
//
// Lo que NO hace:
//   - capturar imágenes (eso es trabajo de CalibrationMode)
//   - geometría estéreo (eso es trabajo de StereoCalibrator)
//   - rectificación o disparidad (fases posteriores)
//
// Uso:
//   MonoCalibrator mc(config);
//   auto result = mc.calibrate(MonoCalibrator::Side::LEFT);
//   if (result.success) mc.saveYAML(result, MonoCalibrator::Side::LEFT);
// ─────────────────────────────────────────────────────────────────────────────
class MonoCalibrator {
public:

    // Qué cámara calibrar en esta ejecución.
    enum class Side { LEFT, RIGHT };

    // ── Struct de resultado ───────────────────────────────────────────────────
    struct Result {
        bool success = false;

        cv::Mat  cameraMatrix;   // matriz intrínseca K de 3×3
        cv::Mat  distCoeffs;     // coeficientes de distorsión [k1,k2,p1,p2,k3]
        cv::Size imageSize;      // tamaño de las imágenes usadas (necesario para stereoCalibrate luego)

        double rpe          = 0.0;  // error de reproyección RMS (menor = mejor)
        int    imagesUsed   = 0;    // imágenes donde se detectaron esquinas
        int    imagesTotal  = 0;    // total de imágenes encontradas en la carpeta

        // Qué archivos de imagen fueron OMITIDOS (detección fallida) — útil para limpieza.
        std::vector<std::string> skippedImages;
    };

    // ─────────────────────────────────────────────────────────────────────────
    explicit MonoCalibrator(const CalibrationConfig& config);

    // Ejecuta el pipeline completo de calibración para un lado.
    // Registra todo en log — no es necesario agregar impresiones alrededor de esta llamada.
    Result calibrate(Side side) const;

    // Guarda el resultado en la ruta YAML definida en CalibrationConfig.
    // Retorna true si tiene éxito.
    bool saveYAML(const Result& result, Side side) const;

    // Imprime un resumen legible por humanos en la salida estándar.
    void printSummary(const Result& result, Side side) const;

private:
    CalibrationConfig  config_;
    ChessboardDetector detector_;

    // ── Funciones auxiliares ──────────────────────────────────────────────────
    std::string imageDir(Side side)  const;
    std::string yamlPath(Side side)  const;
    std::string sideName(Side side)  const;  // "LEFT" o "RIGHT"

    // Construye las posiciones 3D conocidas de las esquinas del tablero.
    // Z = 0 para todas porque el tablero es plano.
    std::vector<cv::Point3f> buildObjectPoints() const;

    // Carga una imagen, la convierte a grises, detecta y refina las esquinas.
    // Retorna true si se encontró el conjunto completo de esquinas.
    bool loadAndDetect(const std::string&         imagePath,
                       std::vector<cv::Point2f>&  outCorners,
                       cv::Size&                  outImageSize) const;

    // Verificación de coherencia del resultado de calibración y advertencia
    // si los valores parecen incorrectos para un sensor OV2640 en resolución VGA.
    void validateResult(const Result& result) const;
};
