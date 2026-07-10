#include "ChessboardDetector.hpp"
#include "DiagnosticLogger.hpp"

ChessboardDetector::ChessboardDetector(const CalibrationConfig& config)
    : config_(config) {}

// ── Funciones auxiliares privadas ────────────────────────────────────────────

double ChessboardDetector::measureBrightness(const cv::Mat& gray) const {
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    return mean[0];  // intensidad media del píxel, rango 0–255
}

double ChessboardDetector::measureContrast(const cv::Mat& gray) const {
    // Contraste de Michelson: (max - min) / (max + min)
    // Rango 0 (gris uniforme) a 1 (blanco y negro puro)
    double minVal, maxVal;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    if (maxVal + minVal < 1.0) return 0.0;
    return (maxVal - minVal) / (maxVal + minVal);
}

cv::Mat ChessboardDetector::applyClahe(const cv::Mat& gray) const {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        config_.claheClipLimit,
        config_.claheTileSize
    );
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);
    return enhanced;
}

// ── Función pública de detección ─────────────────────────────────────────────

DetectionResult ChessboardDetector::detect(const cv::Mat& grayFrame,
                                           cv::Mat&       displayFrame) const {
    DetectionResult result;
    result.cornersExpected = config_.boardSize.width * config_.boardSize.height;

    // ── Paso 1: medir calidad de la imagen ───────────────────────────────────
    result.brightness = measureBrightness(grayFrame);
    result.contrast   = measureContrast(grayFrame);

    // ── Paso 2: decidir si aplicar CLAHE ─────────────────────────────────────
    // CLAHE (Ecualización de Histograma Adaptativa con Límite de Contraste) ecualiza
    // el contraste localmente por mosaico. Ayuda cuando el tablero está oscuro
    // pero el patrón geométrico sigue presente.
    //
    // Trabajamos sobre una COPIA — nunca modificamos el grayFrame del llamante,
    // porque puede necesitar el original para guardarlo en disco.
    cv::Mat workGray = grayFrame;
    if (config_.autoEnhance && result.brightness < config_.brightnessThreshold) {
        workGray         = applyClahe(grayFrame);
        result.claheUsed = true;
    }

    // ── Paso 3: detección gruesa ──────────────────────────────────────────────
    result.found = cv::findChessboardCorners(
        workGray,
        config_.boardSize,
        result.corners,
        config_.findFlags
    );

    // Contamos cuántas esquinas se encontraron aunque el tablero no esté completo.
    // Útil para diagnosticar oclusión parcial vs fallo total.
    result.cornersFound = static_cast<int>(result.corners.size());

    if (!result.found) {
        // Si CLAHE todavía no se intentó y el brillo es marginal (50–80),
        // intentamos una vez más forzando CLAHE.
        if (!result.claheUsed && result.brightness < 100.0) {
            cv::Mat enhanced = applyClahe(grayFrame);
            result.found = cv::findChessboardCorners(
                enhanced,
                config_.boardSize,
                result.corners,
                config_.findFlags
            );
            result.claheUsed    = true;
            result.cornersFound = static_cast<int>(result.corners.size());
        }

        if (!result.found) {
            return result;  // no encontrado — el llamante registrará el evento
        }
    }

    // ── Paso 4: refinamiento sub-píxel ───────────────────────────────────────
    // cornerSubPix refina cada esquina de ~1px de precisión a ~0.01px.
    // Trabaja sobre el mismo workGray usado en la detección (con o sin CLAHE).
    cv::cornerSubPix(
        workGray,
        result.corners,
        config_.subPixWinSize,
        cv::Size(-1, -1),
        config_.subPixCriteria
    );

    // ── Paso 5: dibujar en el frame de visualización ─────────────────────────
    // Se dibuja SOLO en el frame de color para mostrar — nunca en grayFrame ni workGray.
    cv::drawChessboardCorners(
        displayFrame,
        config_.boardSize,
        result.corners,
        result.found
    );

    return result;
}