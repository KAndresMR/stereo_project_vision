#include "Rectifier.hpp"
#include "DiagnosticLogger.hpp"
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
Rectifier::Rectifier(const CalibrationConfig& config)
    : config_(config) {}

// ─────────────────────────────────────────────────────────────────────────────
// loadYAMLs
//
// Lee TODOS los datos de calibración de los tres archivos YAML generados
// por las dos fases anteriores:
//   left.yaml   → K_left, dist_left
//   right.yaml  → K_right, dist_right
//   stereo.yaml → R, T, R1, R2, P1, P2, Q (pre-calculados por StereoCalibrator)
//
// Nota: R1, R2, P1, P2, Q ya están almacenados en stereo.yaml porque
// StereoCalibrator ejecuta stereoRectify antes de guardar. Esto evita
// volver a ejecutar stereoRectify aquí con parámetros potencialmente distintos.
// ─────────────────────────────────────────────────────────────────────────────
bool Rectifier::loadYAMLs(cv::Mat& K_left,  cv::Mat& dist_left,
                           cv::Mat& K_right, cv::Mat& dist_right,
                           cv::Mat& R,       cv::Mat& T,
                           cv::Mat& R1,      cv::Mat& R2,
                           cv::Mat& P1,      cv::Mat& P2,
                           cv::Mat& Q,
                           cv::Size& imageSize) const {
    // ── left.yaml ─────────────────────────────────────────────────────────────
    {
        cv::FileStorage fs(config_.leftYaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("Rectifier", "No se puede abrir: " + config_.leftYaml);
            return false;
        }
        fs["camera_matrix"]           >> K_left;
        fs["distortion_coefficients"] >> dist_left;
        Log::yamlLoaded(config_.leftYaml, true);
    }

    // ── right.yaml ────────────────────────────────────────────────────────────
    {
        cv::FileStorage fs(config_.rightYaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("Rectifier", "No se puede abrir: " + config_.rightYaml);
            return false;
        }
        fs["camera_matrix"]           >> K_right;
        fs["distortion_coefficients"] >> dist_right;
        Log::yamlLoaded(config_.rightYaml, true);
    }

    // ── stereo.yaml ───────────────────────────────────────────────────────────
    {
        cv::FileStorage fs(config_.stereoYaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("Rectifier", "No se puede abrir: " + config_.stereoYaml);
            Log::error("Rectifier", "Ejecute primero la calibracion estereo (opcion 4 del menu).");
            return false;
        }
        fs["R"]  >> R;   fs["T"]  >> T;
        fs["R1"] >> R1;  fs["R2"] >> R2;
        fs["P1"] >> P1;  fs["P2"] >> P2;
        fs["Q"]  >> Q;
        int w = 0, h = 0;
        fs["image_width"] >> w; fs["image_height"] >> h;
        if (w > 0 && h > 0) imageSize = {w, h};
        Log::yamlLoaded(config_.stereoYaml, true);
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute
//
// Construye las tablas de remapeo de píxeles (map1x, map1y, map2x, map2y).
//
// Cómo funciona el remapeo:
//   Para cada píxel de salida (u, v) en la imagen rectificada, la tabla de remap
//   almacena el píxel fuente (u', v') en la imagen original distorsionada.
//   cv::remap() luego muestrea la imagen original en (u', v') usando interpolación
//   bilineal para rellenar (u, v) en la salida.
//
//   Este es un cálculo único: las tablas se almacenan y se reutilizan en
//   cada frame en tiempo real (muy rápido — solo una búsqueda por píxel).
// ─────────────────────────────────────────────────────────────────────────────
bool Rectifier::compute() {
    Log::separator("Rectifier — calculando tablas de remapeo");

    cv::Mat K_left, dist_left, K_right, dist_right, R, T;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imageSize;

    if (!loadYAMLs(K_left, dist_left, K_right, dist_right,
                   R, T, R1, R2, P1, P2, Q, imageSize)) {
        return false;
    }

    maps_.R1 = R1;  maps_.R2 = R2;
    maps_.P1 = P1;  maps_.P2 = P2;
    maps_.Q  = Q;
    maps_.imageSize = imageSize;

    // ── initUndistortRectifyMap ───────────────────────────────────────────────
    //
    // Combina dos operaciones en una sola tabla de remap:
    //   1. Corrección de distorsión: elimina la distorsión del lente usando K y dist
    //   2. Rectificación: aplica R1 (o R2) para alinear las líneas epipolares
    //
    // CV_32FC1: mapa en punto flotante — precisión sub-píxel en la búsqueda.
    //
    cv::initUndistortRectifyMap(
        K_left, dist_left, R1, P1,
        imageSize, CV_32FC1,
        maps_.map1x, maps_.map1y
    );

    cv::initUndistortRectifyMap(
        K_right, dist_right, R2, P2,
        imageSize, CV_32FC1,
        maps_.map2x, maps_.map2y
    );

    maps_.ready = true;

    Log::info("Rectifier", "Tablas de remapeo calculadas para " +
              std::to_string(imageSize.width) + "×" +
              std::to_string(imageSize.height));
    Log::info("Rectifier",
        "Q[3][2] = " + std::to_string(-1.0 / Q.at<double>(3,2)) +
        " m (baseline codificado)");
    Log::info("Rectifier",
        "Q[2][3] = " + std::to_string(Q.at<double>(2,3)) +
        " px (longitud focal codificada)");

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// rectify
//
// Aplica las tablas de remapeo pre-calculadas a un par estéreo sin procesar.
// Los frames de entrada pueden ser del disco o del stream en vivo.
//
// cv::INTER_LINEAR: interpolación bilineal — buen balance entre calidad y velocidad.
// cv::INTER_LANCZOS4: mayor calidad pero más lento (usar para depuración/offline).
// ─────────────────────────────────────────────────────────────────────────────
std::pair<cv::Mat, cv::Mat> Rectifier::rectify(const cv::Mat& rawLeft,
                                                const cv::Mat& rawRight) const {
    if (!maps_.ready) {
        Log::error("Rectifier", "Tablas no calculadas. Llame primero a compute().");
        return {};
    }

    cv::Mat rectLeft, rectRight;
    cv::remap(rawLeft,  rectLeft,  maps_.map1x, maps_.map1y, cv::INTER_LINEAR);
    cv::remap(rawRight, rectRight, maps_.map2x, maps_.map2y, cv::INTER_LINEAR);

    return {rectLeft, rectRight};
}

// ─────────────────────────────────────────────────────────────────────────────
// drawEpipolarLines
//
// Crea una imagen compuesta lado a lado con líneas horizontales verdes.
//
// Cómo interpretar el resultado:
//   BUENO: una característica distintiva (esquina, borde) en la imagen izquierda
//          en la fila Y aparece en la MISMA fila Y en la imagen derecha.
//   MALO:  la misma característica aparece en filas diferentes → la rectificación
//          falló o la calibración fue deficiente.
//
// Causas típicas de mala alineación:
//   - stereo.yaml calculado con un tamaño de imagen diferente al de los frames actuales
//   - las cámaras se movieron físicamente después de la calibración
//   - calibración estéreo deficiente (RMS alto o muy pocos pares)
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat Rectifier::drawEpipolarLines(const cv::Mat& rectLeft,
                                      const cv::Mat& rectRight,
                                      int lineSpacing) const {
    cv::Mat dispL = rectLeft.clone();
    cv::Mat dispR = rectRight.clone();

    // Imagen combinada lado a lado
    cv::Mat combined;
    cv::hconcat(dispL, dispR, combined);

    // Dibujar líneas epipolares horizontales
    const cv::Scalar lineColor(0, 220, 0);  // verde
    for (int y = lineSpacing; y < combined.rows; y += lineSpacing) {
        cv::line(combined, {0, y}, {combined.cols, y}, lineColor, 1);
        // Número de fila (cada dos líneas para no saturar la pantalla)
        if ((y / lineSpacing) % 2 == 0) {
            cv::putText(combined, std::to_string(y),
                        {2, y - 3}, cv::FONT_HERSHEY_SIMPLEX,
                        0.35, lineColor, 1);
        }
    }

    // Etiquetas de las cámaras
    cv::putText(combined, "IZQ (rectificada)",
                {10, 18}, cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 220, 255), 1);
    cv::putText(combined, "DER (rectificada)",
                {rectLeft.cols + 10, 18}, cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 220, 255), 1);
    cv::putText(combined, "Las lineas deben cruzar los mismos elementos en ambas imagenes",
                {10, combined.rows - 8}, cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(180, 180, 180), 1);

    return combined;
}

// ─────────────────────────────────────────────────────────────────────────────
// previewDataset
//
// Validación offline: carga cada par estéreo guardado, lo rectifica y lo muestra
// con líneas epipolares. Útil para comprobar la calidad de la calibración
// antes de conectar las cámaras en vivo.
// ─────────────────────────────────────────────────────────────────────────────
void Rectifier::previewDataset() const {
    if (!maps_.ready) {
        Log::error("Rectifier", "Llame a compute() antes de previewDataset().");
        return;
    }

    std::vector<std::string> leftPaths, rightPaths;
    cv::glob(config_.datasetDir + "/left/*.jpg",  leftPaths,  false);
    cv::glob(config_.datasetDir + "/right/*.jpg", rightPaths, false);
    std::sort(leftPaths.begin(),  leftPaths.end());
    std::sort(rightPaths.begin(), rightPaths.end());

    int n = static_cast<int>(std::min(leftPaths.size(), rightPaths.size()));
    if (n == 0) {
        Log::error("Rectifier", "No se encontraron pares de imagenes en el dataset.");
        return;
    }

    Log::info("Rectifier",
        "Vista previa offline: " + std::to_string(n) +
        " pares. CUALQUIER TECLA = siguiente | ESC = salir");

    for (int i = 0; i < n; ++i) {
        cv::Mat rawL = cv::imread(leftPaths[i],  cv::IMREAD_COLOR);
        cv::Mat rawR = cv::imread(rightPaths[i], cv::IMREAD_COLOR);
        if (rawL.empty() || rawR.empty()) continue;

        auto [rectL, rectR] = rectify(rawL, rawR);
        cv::Mat debug = drawEpipolarLines(rectL, rectR);

        // Agregar etiqueta con el índice del par
        std::string label = "Par " + std::to_string(i+1) + "/" +
                            std::to_string(n) + "  — " +
                            std::filesystem::path(leftPaths[i]).filename().string();
        cv::putText(debug, label,
                    {debug.cols/2 - 180, 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1);

        // Escalar para visualización si es demasiado ancho
        double scale = std::min(1.0, 1400.0 / debug.cols);
        if (scale < 1.0) cv::resize(debug, debug, {}, scale, scale);

        cv::imshow("Verificacion Epipolar", debug);
        int key = cv::waitKey(0);
        if (key == 27) break;  // ESC
    }

    cv::destroyWindow("Verificacion Epipolar");
}