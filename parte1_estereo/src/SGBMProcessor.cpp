#include "SGBMProcessor.hpp"
#include "DiagnosticLogger.hpp"
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
SGBMProcessor::SGBMProcessor(const Params& params) : params_(params) {
    init();
}

// ─────────────────────────────────────────────────────────────────────────────
// init — (re)crea los matchers con los params_ actuales
//
// Se llama en la construcción y tras setParams().
// Costoso (~1ms) pero no por frame — solo al cambiar parámetros.
// ─────────────────────────────────────────────────────────────────────────────
void SGBMProcessor::init() {
    const int ch = 1;
    const int bs = params_.blockSize;

    int p1 = params_.P1;
    int p2 = params_.P2;
    if (p1 == 0 || p2 == 0) {
        // Fórmula matemática de OpenCV para penalización dinámica
        p1 = 8 * 1 * params_.blockSize * params_.blockSize;
        p2 = 32 * 1 * params_.blockSize * params_.blockSize;
    }

    leftMatcher_ = cv::StereoSGBM::create(
        params_.minDisparity,
        params_.numDisparities,
        params_.blockSize,
        p1, p2,
        params_.disp12MaxDiff,
        params_.preFilterCap,
        params_.uniquenessRatio,
        params_.speckleWindowSize,
        params_.speckleRange,
        params_.mode
    );

    // Matcher derecho: espejo del izquierdo en dirección derecha→izquierda.
    // Requerido para el filtro WLS. Barato de crear.
    if (params_.useWLS) {
        rightMatcher_ = cv::ximgproc::createRightMatcher(leftMatcher_);
        wlsFilter_    = cv::ximgproc::createDisparityWLSFilter(leftMatcher_);
        wlsFilter_->setLambda(params_.wlsLambda);
        wlsFilter_->setSigmaColor(params_.wlsSigma);
        Log::info("SGBM", "Filtro WLS activado (λ=" +
                  std::to_string((int)params_.wlsLambda) +
                  " σ=" + std::to_string(params_.wlsSigma) + ")");
    }

    // CLAHE para pre-procesamiento: mejora SGBM en escenas de bajo contraste.
    clahe_ = cv::createCLAHE();

    // Reiniciar buffer temporal (los parámetros pueden haber cambiado el tamaño de ventana)
    smoothed_      = cv::Mat{};
    smoothedValid_ = false;

    Log::info("SGBM", "Inicializado: numDisp=" +
              std::to_string(params_.numDisparities) +
              " blockSize=" + std::to_string(params_.blockSize) +
              " P1=" + std::to_string(p1) +
              " P2=" + std::to_string(p2));
}

// ─────────────────────────────────────────────────────────────────────────────
void SGBMProcessor::setParams(const Params& p) {
    params_ = p;
    init();
}

// ─────────────────────────────────────────────────────────────────────────────
// compute — Paso 1
//
// Aplica CLAHE a las imágenes rectificadas en escala de grises y luego ejecuta StereoSGBM.
// Si WLS está activado, también ejecuta el matcher derecho aquí para que postprocess()
// pueda usar rightDisp_ sin recalcular.
//
// Salida: mapa de disparidad CV_16S. Disparidad real = valor / 16.0 píxeles.
// Los píxeles inválidos tienen valor < params_.minDisparity * 16.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::compute(const cv::Mat& rectLeft, const cv::Mat& rectRight) {
    // Convertir a escala de grises (SGBM trabaja internamente en grises,
    // pero aceptar entrada en color es más flexible para el llamante)
    cv::Mat grayL, grayR;
    if (rectLeft.channels() == 3) {
        cv::cvtColor(rectLeft,  grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rectRight, grayR, cv::COLOR_BGR2GRAY);
    } else {
        grayL = rectLeft;
        grayR = rectRight;
    }

    clahe_->setClipLimit(params_.claheClipLimit);
    clahe_->setTilesGridSize(cv::Size(std::max(2, params_.claheTileSize),
                                      std::max(2, params_.claheTileSize)));

    cv::Mat enhL, enhR;
    if (params_.preBlurSigma > 0.1) {
        cv::GaussianBlur(grayL, grayL, cv::Size(3,3), params_.preBlurSigma);
        cv::GaussianBlur(grayR, grayR, cv::Size(3,3), params_.preBlurSigma);
    }
    clahe_->apply(grayL, enhL);
    clahe_->apply(grayR, enhR);
    enhancedL_ = enhL.clone(); // guardado como guía para WLS y visualización en dashboard

    // Disparidad izquierda (salida principal)
    cv::Mat leftDisp;
    leftMatcher_->compute(enhL, enhR, leftDisp);

    // Disparidad derecha (solo se necesita para WLS; se guarda para postprocess())
    if (params_.useWLS && rightMatcher_) {
        rightMatcher_->compute(enhR, enhL, rightDisp_);
    }

    return leftDisp;
}

// ─────────────────────────────────────────────────────────────────────────────
// postprocess — Paso 2: filtro WLS
//
// El filtro WLS (Mínimos Cuadrados Ponderados) resuelve:
//
//   argmin_u { ||u - d||² + λ * Σ w_ij(I) * (u_i - u_j)² }
//
// donde d = disparidad cruda, w_ij(I) = peso con conciencia de bordes de la imagen I.
//
// En lenguaje simple:
//   - Rellena huecos (donde SGBM no encontró correspondencia) difundiendo valores
//     vecinos, guiado por los bordes de la imagen (no difumina sobre bordes de profundidad reales).
//   - λ controla la suavidad: mayor valor → más suave pero menos detalle.
//   - σ controla la sensibilidad a bordes: mayor valor → difunde más a través de bordes.
//
// ¿Por qué se necesita la disparidad derecha?
//   WLS usa la disparidad derecha para calcular un mapa de confianza: un píxel es
//   "confiable" si las disparidades izquierda→derecha y derecha→izquierda coinciden.
//   Los píxeles no confiables son rellenados por el filtro en lugar de pasarse tal cual.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::postprocess(const cv::Mat& rawDisparity,
                                    const cv::Mat& rectLeft) {
    if (!params_.useWLS || !wlsFilter_ || rightDisp_.empty()) {
        return rawDisparity;
    }
    cv::Mat filtered;
    wlsFilter_->filter(
        rawDisparity,
        enhancedL_, // Usamos la imagen con CLAHE como guía
        filtered,
        rightDisp_
    );
    return filtered;
}

// ─────────────────────────────────────────────────────────────────────────────
// temporalSmooth — Paso 3: Media Móvil Exponencial (EMA)
//
// Fórmula EMA: S(t) = α * D(t) + (1-α) * S(t-1)
//
// α = temporalAlpha:
//   - Alto (0.8–1.0): respuesta rápida, suavizado mínimo.
//   - Bajo (0.1–0.3): suavizado intenso, efecto fantasma en objetos en movimiento.
//   - 0.4 es un buen valor por defecto para escenas mayormente estáticas.
//
// Opera sobre CV_32F para evitar errores de redondeo en la acumulación.
// Los píxeles inválidos (<=0) en el frame actual NO se actualizan,
// preservando la última estimación válida — evita difuminar regiones inválidas.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::temporalSmooth(const cv::Mat& filteredDisparity) {
    if (!params_.useTemporalSmoothing) return filteredDisparity;

    cv::Mat current32f;
    filteredDisparity.convertTo(current32f, CV_32F);

    if (!smoothedValid_ || smoothed_.size() != current32f.size()) {
        smoothed_      = current32f.clone();
        smoothedValid_ = true;
        return filteredDisparity;
    }

    // Solo actualizar píxeles que tienen disparidad válida en el frame actual.
    // Inválido = disparidad <= minDisparity * 16 (convención SGBM).
    cv::Mat validMask = (filteredDisparity > params_.minDisparity * 16);

    // EMA solo en píxeles válidos
    float alpha = params_.temporalAlpha;
    cv::Mat blended;

    cv::addWeighted(
        current32f,
        alpha,
        smoothed_,
        1.0f - alpha,
        0.0,
        blended);

    // Actualizar SOLO los píxeles válidos
    blended.copyTo(smoothed_, validMask);

    // Escribir de vuelta a píxeles inválidos (mantener la última estimación válida allí)
    // — este es el comportamiento de "relleno temporal"
    cv::Mat result;
    smoothed_.convertTo(result, CV_16S);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// visualize — Paso 4: disparidad 16S → imagen en falso color de 8 bits
//
// SGBM emite CV_16S donde cada valor = disparidad_real * 16.
// Dividimos por 16, recortamos negativos (inválidos), escalamos a 0–255.
// COLORMAP_TURBO: colores cálidos = cerca, colores fríos = lejos.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::visualize(const cv::Mat& disparity) const {
    // Convertir a float, dividir por 16 para obtener la disparidad real en píxeles
    cv::Mat disp32f;
    disparity.convertTo(disp32f, CV_32F, 1.0 / 16.0);

    // Recortar negativos e inválidos (píxeles < minDisparity*16 en 16S = inválidos en SGBM)
    cv::Mat invalid = (disparity <= params_.minDisparity * 16);
    cv::threshold(disp32f, disp32f, 0.0, 0.0, cv::THRESH_TOZERO);

    // Normalización RELATIVA: lo más lejano válido → 0 (azul), lo más cercano → 255 (rojo)
    // Esto asegura que el objeto más cercano en escena siempre se vea rojo,
    // independientemente de las distancias absolutas.
    cv::Mat disp8u;
    cv::normalize(disp32f, disp8u, 0, 255, cv::NORM_MINMAX, CV_8U, ~invalid);

    // Poner en negro los píxeles inválidos ANTES del mapeo de color
    disp8u.setTo(0, invalid);

    cv::Mat colored;
    cv::applyColorMap(disp8u, colored, cv::COLORMAP_TURBO);

    // Poner en negro los píxeles inválidos también en la imagen coloreada
    colored.setTo(cv::Scalar(0, 0, 0), invalid);

    return colored;
}

// ─────────────────────────────────────────────────────────────────────────────
// toDepth — Disparidad → profundidad real en metros via matriz Q
//
// La matriz Q de stereoRectify codifica:
//   Z = f * B / d
// donde f = distancia focal, B = baseline, d = disparidad.
//
// reprojectImageTo3D calcula (X, Y, Z) para cada píxel.
// Solo extraemos Z (profundidad) y aplicamos recorte de rango.
//
// Píxeles inválidos: disparidad <= 0 tras conversión → profundidad = 0.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::toDepth(const cv::Mat& disparity,
                                const cv::Mat& Q) const {
    // Convertir disparidad a float (disparidad real en píxeles)
    cv::Mat disp32f;
    disparity.convertTo(disp32f, CV_32F, 1.0 / 16.0);

    // reprojectImageTo3D: para cada píxel, calcula (X, Y, Z) en coordenadas de cámara
    // handleMissingValues=true: pone los píxeles fuera de rango en 10000
    cv::Mat points3D;
    cv::reprojectImageTo3D(disp32f, points3D, Q, true);

    // Extraer canal Z (profundidad en metros)
    cv::Mat channels[3];
    cv::split(points3D, channels);
    cv::Mat depthM = channels[2];

    // Máscara: disparidad válida Y profundidad en rango esperado
    cv::Mat validDisp  = (disp32f > 0.0f);
    cv::Mat validDepth = (depthM > params_.minDepthM) &
                         (depthM < params_.maxDepthM);
    cv::Mat valid = validDisp & validDepth;

    // Poner en cero los píxeles inválidos
    depthM.setTo(0.0f, ~valid);
    return depthM;  // CV_32F, en metros
}

// ─────────────────────────────────────────────────────────────────────────────
// visualizeDepth — mapa de profundidad → imagen en falso color para visualización
//
// Escala lineal: minDepthM → azul (lejos), maxDepthM → rojo (cerca).
// Los píxeles inválidos (cero) se muestran en negro.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat SGBMProcessor::visualizeDepth(const cv::Mat& depthM) const {
    cv::Mat normalized;
    double scale = 255.0 / (params_.maxDepthM - params_.minDepthM);
    depthM.convertTo(normalized, CV_8U, scale,
                     -params_.minDepthM * scale);

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);

    // Poner en negro los píxeles inválidos (profundidad == 0)
    colored.setTo(cv::Scalar(0,0,0), depthM == 0.0f);

    return colored;
}