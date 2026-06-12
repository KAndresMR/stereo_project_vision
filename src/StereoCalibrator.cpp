#include "StereoCalibrator.hpp"
#include "DiagnosticLogger.hpp"
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
StereoCalibrator::StereoCalibrator(const CalibrationConfig& config)
    : config_(config)
    , detector_(config) {}

// ─────────────────────────────────────────────────────────────────────────────
// buildObjectPoints — construye la cuadrícula 3D perfecta del tablero
// ─────────────────────────────────────────────────────────────────────────────
std::vector<cv::Point3f> StereoCalibrator::buildObjectPoints() const {
    std::vector<cv::Point3f> pts;
    pts.reserve(config_.boardSize.width * config_.boardSize.height);
    for (int r = 0; r < config_.boardSize.height; ++r)
        for (int c = 0; c < config_.boardSize.width; ++c)
            pts.push_back({c * config_.squareSizeM,
                           r * config_.squareSizeM,
                           0.0f});
    return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// loadIntrinsics
//
// Lee K y distCoeffs para cada cámara desde los archivos YAML individuales
// generados por MonoCalibrator.
// ─────────────────────────────────────────────────────────────────────────────
bool StereoCalibrator::loadIntrinsics(cv::Mat& K_left,  cv::Mat& dist_left,
                                      cv::Mat& K_right, cv::Mat& dist_right,
                                      cv::Size& imageSize) const {
    auto load = [&](const std::string& path,
                    cv::Mat& K, cv::Mat& dist,
                    const std::string& name) -> bool {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("StereoCalib", "No se puede abrir: " + path);
            Log::error("StereoCalib", "Ejecute primero la calibracion monocular (opcion 3 del menu).");
            return false;
        }
        fs["camera_matrix"]           >> K;
        fs["distortion_coefficients"] >> dist;
        int w = 0, h = 0;
        fs["image_width"]  >> w;
        fs["image_height"] >> h;
        if (w > 0 && h > 0) imageSize = {w, h};
        fs.release();
        Log::yamlLoaded(path, true);
        Log::info("StereoCalib",
            name + ": fx=" + std::to_string((int)K.at<double>(0,0)) +
            " fy=" + std::to_string((int)K.at<double>(1,1)) +
            " cx=" + std::to_string((int)K.at<double>(0,2)) +
            " cy=" + std::to_string((int)K.at<double>(1,2)));
        return true;
    };

    return load(config_.leftYaml,  K_left,  dist_left,  "IZQ ")
        && load(config_.rightYaml, K_right, dist_right, "DER ");
}

// ─────────────────────────────────────────────────────────────────────────────
// detectPair
//
// Carga un par estéreo del disco y detecta las esquinas en AMBAS imágenes.
// Solo retorna true cuando AMBAS cámaras encuentran el patrón completo.
// Un par donde solo una cámara tiene éxito es inútil para stereoCalibrate
// porque necesitamos puntos 2D correspondientes en izquierda Y derecha.
// ─────────────────────────────────────────────────────────────────────────────
bool StereoCalibrator::detectPair(const std::string& leftPath,
                                  const std::string& rightPath,
                                  std::vector<cv::Point2f>& cornersL,
                                  std::vector<cv::Point2f>& cornersR,
                                  cv::Size& imageSize) const {
    cv::Mat imgL = cv::imread(leftPath,  cv::IMREAD_COLOR);
    cv::Mat imgR = cv::imread(rightPath, cv::IMREAD_COLOR);

    if (imgL.empty() || imgR.empty()) {
        Log::error("StereoCalib", "No se puede leer el par: " +
                   fs::path(leftPath).filename().string());
        return false;
    }

    if (imgL.size() != imgR.size()) {
        Log::warn("StereoCalib",
            "Tamaños de imagen diferentes entre IZQUIERDA y DERECHA.");
        return false;
    }

    imageSize = imgL.size();

    cv::Mat grayL, grayR;
    cv::cvtColor(imgL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, grayR, cv::COLOR_BGR2GRAY);

    cv::Mat dummyL = imgL, dummyR = imgR;
    DetectionResult dL = detector_.detect(grayL, dummyL);
    DetectionResult dR = detector_.detect(grayR, dummyR);

    std::string fname = fs::path(leftPath).filename().string();

    if (dL.found && dR.found) {
        cornersL = dL.corners;
        cornersR = dR.corners;
        Log::info("StereoCalib",
            "  ✓ " + fname +
            " | brillo_IZQ=" + std::to_string((int)dL.brightness) +
            " brillo_DER=" + std::to_string((int)dR.brightness));
        return true;
    }

    // Indicar cuál lado falló
    std::string razon;
    if (!dL.found && !dR.found) razon = "AMBAS camaras fallaron";
    else if (!dL.found)         razon = "IZQUIERDA fallo (esquinas=" +
                                         std::to_string(dL.cornersFound) + "/"+
                                         std::to_string(dL.cornersExpected)+")";
    else                         razon = "DERECHA fallo (esquinas=" +
                                         std::to_string(dR.cornersFound) + "/"+
                                         std::to_string(dR.cornersExpected)+")";

    Log::warn("StereoCalib", "  ✗ " + fname + " | " + razon + " → OMITIDO");
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// validateResult — auditoría de coherencia física del resultado
// ─────────────────────────────────────────────────────────────────────────────
void StereoCalibrator::validateResult(const Result& result) const {
    const std::string tag = "Validacion";

    // ── Verificar Baseline ─────────────────────────────────────────────────────────────
    double blMm = result.baselineM * 1000.0;
    Log::info(tag, "Baseline: " + std::to_string(blMm) + " mm");
    if (result.baselineM < 0.03)
        Log::warn(tag, "Baseline < 30mm — las camaras pueden estar muy juntas o T esta mal");
    else if (result.baselineM > 0.20)
        Log::warn(tag, "Baseline > 200mm — inusualmente grande. Revise la posicion de las camaras.");
    else
        Log::info(tag, "Baseline en rango esperado ✓");

    // ── RMS ──────────────────────────────────────────────────────────────────
    if      (result.rpe < 0.3)  Log::info(tag, "RMS=" + std::to_string(result.rpe) + " ✓✓ Excelente");
    else if (result.rpe < 0.5)  Log::info(tag, "RMS=" + std::to_string(result.rpe) + " ✓  Bueno");
    else if (result.rpe < 1.0)  Log::warn(tag, "RMS=" + std::to_string(result.rpe) + " △  Aceptable");
    else                        Log::warn(tag, "RMS=" + std::to_string(result.rpe) + " ✗  Deficiente — se recomienda recapturar");

    // ── Verificar dirección del vector T ────────────────────────────────────────────────
    // T[0] debe dominar (cámaras separadas horizontalmente).
    // T[1] y T[2] deben ser fracciones pequeñas de T[0].
    double tx = std::abs(result.T.at<double>(0));
    double ty = std::abs(result.T.at<double>(1));
    double tz = std::abs(result.T.at<double>(2));
    if (ty > tx * 0.3 || tz > tx * 0.3)
        Log::warn(tag, "T tiene componente vertical/profundidad significativa — las camaras pueden no ser coplanares");
    else
        Log::info(tag, "Direccion de T correcta (predominantemente horizontal) ✓");

    // ── Verificar ángulo de rotación entre cámaras ──────────────────────────────────
    cv::Mat rvec;
    cv::Rodrigues(result.R, rvec);
    double angleRad = cv::norm(rvec);
    double angleDeg = angleRad * 180.0 / CV_PI;
    Log::info(tag, "Angulo de rotacion entre camaras: " + std::to_string(angleDeg) + " grados");
    if (angleDeg > 5.0)
        Log::warn(tag, "Rotacion > 5 grados entre camaras — desalineacion fisica grande");

    // ── Pares utilizados ──────────────────────────────────────────────────────
    if (result.pairsUsed < 10)
        Log::warn(tag, "Solo se usaron " + std::to_string(result.pairsUsed) +
                  " pares — apunte a 15 o mas");
}

// ─────────────────────────────────────────────────────────────────────────────
// calibrate — el pipeline principal
// ─────────────────────────────────────────────────────────────────────────────
StereoCalibrator::Result StereoCalibrator::calibrate() const {
    Result result;

    Log::separator("StereoCalibrator — Fase 2");

    // ── Paso 1: cargar intrínsecos individuales ───────────────────────────────
    cv::Mat K_left, dist_left, K_right, dist_right;
    cv::Size imageSize;

    if (!loadIntrinsics(K_left, dist_left, K_right, dist_right, imageSize)) {
        return result;
    }

    // ── Paso 2: encontrar pares de imágenes correspondientes ─────────────────
    std::vector<std::string> leftPaths, rightPaths;
    cv::glob(config_.datasetDir + "/left/*.jpg",  leftPaths,  false);
    cv::glob(config_.datasetDir + "/right/*.jpg", rightPaths, false);
    std::sort(leftPaths.begin(),  leftPaths.end());
    std::sort(rightPaths.begin(), rightPaths.end());

    result.pairsTotal = static_cast<int>(
        std::min(leftPaths.size(), rightPaths.size()));

    Log::info("StereoCalib",
        "Pares de imagenes encontrados: " + std::to_string(result.pairsTotal));

    if (result.pairsTotal == 0) {
        Log::error("StereoCalib", "No se encontraron pares de imagenes. Ejecute primero la captura.");
        return result;
    }

    // ── Paso 3: detectar esquinas en cada par sincronizado ────────────────────
    Log::info("StereoCalib", "Detectando esquinas en pares estereo...");

    const auto singleObjPts  = buildObjectPoints();
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imgPtsL, imgPtsR;

    for (int i = 0; i < result.pairsTotal; ++i) {
        std::vector<cv::Point2f> cL, cR;
        cv::Size sz;

        if (detectPair(leftPaths[i], rightPaths[i], cL, cR, sz)) {
            if (imageSize.empty()) imageSize = sz;
            objectPoints.push_back(singleObjPts);
            imgPtsL.push_back(cL);
            imgPtsR.push_back(cR);
        }
    }

    result.pairsUsed = static_cast<int>(objectPoints.size());
    Log::info("StereoCalib",
        "Pares estereo validos: " + std::to_string(result.pairsUsed) +
        " / " + std::to_string(result.pairsTotal));

    if (result.pairsUsed < 6) {
        Log::error("StereoCalib",
            "Se necesitan al menos 6 pares validos, se obtuvieron: " +
            std::to_string(result.pairsUsed));
        return result;
    }

    // ── Paso 4: stereoCalibrate ───────────────────────────────────────────────
    //
    // Ya los calibramos individualmente. Aquí solo resolvemos R y T
    // (la transformación rígida de 6-DoF de la cámara izquierda a la derecha).
    //
    // TermCriteria: permite hasta 100 iteraciones o hasta que el cambio sea < 1e-6.
    //
    Log::info("StereoCalib", "Ejecutando cv::stereoCalibrate...");

    try {
        result.rpe = cv::stereoCalibrate(
            objectPoints,
            imgPtsL, imgPtsR,
            K_left,  dist_left,
            K_right, dist_right,
            imageSize,
            result.R, result.T,
            result.E, result.F,
            cv::CALIB_FIX_INTRINSIC,
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                100, 1e-6)
        );
    } catch (const cv::Exception& e) {
        Log::error("StereoCalib", "stereoCalibrate lanzo excepcion: " +
                   std::string(e.what()));
        return result;
    }

    // ── Paso 5: calcular el Baseline ─────────────────────────────────────────
    // La norma Euclidiana de T = sqrt(Tx² + Ty² + Tz²) = distancia real entre cámaras
    result.baselineM = cv::norm(result.T);
    result.imageSize = imageSize;
    result.success   = true;

    validateResult(result);
    printSummary(result);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Guarda R, T, E, F de stereoCalibrate, más R1, R2, P1, P2, Q 
// ─────────────────────────────────────────────────────────────────────────────
bool StereoCalibrator::saveYAML(const Result& result) const {
    if (!result.success) {
        Log::error("StereoCalib", "No se puede guardar — la calibracion no fue exitosa.");
        return false;
    }

    // ── Recargar intrínsecos para ejecutar stereoRectify ─────────────────────
    cv::Mat K_left, dist_left, K_right, dist_right;
    cv::Size imageSize;
    if (!loadIntrinsics(K_left, dist_left, K_right, dist_right, imageSize)) {
        Log::error("StereoCalib", "No se pueden recargar los intrinsecos para la rectificacion.");
        return false;
    }
    if (imageSize.empty()) imageSize = result.imageSize;

    // ── stereoRectify ─────────────────────────────────────────────────────────
    //
    // Calcula las matrices de rotación (R1, R2) y proyección (P1, P2)
    // que hacen que las líneas epipolares sean horizontales y coplanares.
    // Q es la matriz de reproyección disparidad→profundidad de 4×4 usada luego:
    //   Z = f * B / d   está codificada en Q.
    //
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(
        K_left,  dist_left,
        K_right, dist_right,
        imageSize,
        result.R, result.T,
        R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY,   // alinear centros ópticos horizontalmente
        0,                           // alpha=0: recortar a región válida
        imageSize                    // tamaño de imagen de salida = tamaño de entrada
    );

    // ── Escribir YAML ─────────────────────────────────────────────────────────
    fs::create_directories(fs::path(config_.stereoYaml).parent_path());

    cv::FileStorage fs(config_.stereoYaml, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        Log::error("StereoCalib", "No se puede escribir: " + config_.stereoYaml);
        return false;
    }

    fs << "image_width"            << result.imageSize.width
       << "image_height"           << result.imageSize.height
       << "pairs_used"             << result.pairsUsed
       << "pairs_total"            << result.pairsTotal
       << "rms_stereo"             << result.rpe
       << "baseline_m"             << result.baselineM;

    // Geometría estéreo
    fs << "R"  << result.R
       << "T"  << result.T
       << "E"  << result.E
       << "F"  << result.F;

    // Matrices de rectificación (necesarias para Rectifier)
    fs << "R1" << R1  << "R2" << R2
       << "P1" << P1  << "P2" << P2
       << "Q"  << Q;

    fs.release();

    Log::yamlSaved(config_.stereoYaml);
    Log::info("StereoCalib",
        "Matriz Q guardada — baseline codificado como: " +
        std::to_string(result.baselineM * 1000.0) + " mm");

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// printSummary — imprime resumen de resultados en consola
// ─────────────────────────────────────────────────────────────────────────────
void StereoCalibrator::printSummary(const Result& result) const {
    if (!result.success) return;

    Log::separator("Resultado — Calibracion Estereo");

    double tx = result.T.at<double>(0) * 1000.0;
    double ty = result.T.at<double>(1) * 1000.0;
    double tz = result.T.at<double>(2) * 1000.0;
    cv::Mat rvec;
    cv::Rodrigues(result.R, rvec);
    double angleDeg = cv::norm(rvec) * 180.0 / CV_PI;

    std::cout << std::fixed << std::setprecision(2)
        << "\n  Traslacion T (mm):    Tx=" << tx << "  Ty=" << ty << "  Tz=" << tz << "\n"
        << "  Baseline             : " << result.baselineM * 1000.0 << " mm\n"
        << "  Rotacion entre cam   : " << std::setprecision(3) << angleDeg << " grados\n"
        << "  Error RMS estereo    : " << std::setprecision(4) << result.rpe << " px\n"
        << "  Pares utilizados     : " << result.pairsUsed << " / " << result.pairsTotal << "\n\n";
}