#include "DisparityMode.hpp"
#include "Rectifier.hpp"
#include "SGBMProcessor.hpp"
#include "Kalman1D.hpp"
#include "DiagnosticLogger.hpp"
#include "DebugViews.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// DashboardState — Estado del zoom de la UI
// ─────────────────────────────────────────────────────────────────────────────
struct DashboardState {
    int activeTile = -1; // -1: vista de cuadricula, 0..5: vista expandida del panel
};

// Callback del mouse para alternar entre vista expandida y de cuadrícula
static void onDashboardMouse(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* state = static_cast<DashboardState*>(userdata);
    
    if (state->activeTile == -1) {
        // Modo cuadricula: calcular qué panel de 320x240 se clickeó
        int col = x / 320;
        int row = y / 240;
        if (col >= 0 && col < 3 && row >= 0 && row < 2) {
            state->activeTile = row * 3 + col;
        }
    } else {
        // Modo expandido: hacer click en cualquier parte para volver a la cuadricula
        state->activeTile = -1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Funciones auxiliares para el HUD
// ─────────────────────────────────────────────────────────────────────────────

// Dibuja una franja semitransparente de cabecera en la parte superior de un panel
static void drawTileHeader(cv::Mat& img, const std::string& title, cv::Scalar accentColor) {
    cv::Rect strip(0, 0, img.cols, 26);
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, strip, cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.7, img, 0.3, 0, img);
    
    cv::line(img, {0, 0}, {0, 25}, accentColor, 3);
    cv::line(img, {0, 25}, {img.cols, 25}, cv::Scalar(40, 40, 40), 1);
    cv::putText(img, title, {10, 17}, cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

// Dibuja una caja de texto detallada semitransparente para los valores del HUD
static void drawHUDTextBox(cv::Mat& img, const std::vector<std::string>& lines, cv::Point topLeft, cv::Scalar color) {
    int font = cv::FONT_HERSHEY_DUPLEX;
    double scale = 0.45;
    int thickness = 1;
    int lineSpacing = 18;
    int padding = 8;
    
    int maxWidth = 0;
    for (const auto& line : lines) {
        int baseline = 0;
        cv::Size s = cv::getTextSize(line, font, scale, thickness, &baseline);
        if (s.width > maxWidth) maxWidth = s.width;
    }
    int totalHeight = lines.size() * lineSpacing + padding;
    
    cv::Rect bgRect(topLeft.x, topLeft.y, maxWidth + padding * 2, totalHeight);
    bgRect &= cv::Rect(0, 0, img.cols, img.rows);
    if (bgRect.width > 0 && bgRect.height > 0) {
        cv::Mat overlay = img.clone();
        cv::rectangle(overlay, bgRect, cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.7, img, 0.3, 0, img);
        cv::rectangle(img, bgRect, color, 1, cv::LINE_AA);
        
        for (size_t i = 0; i < lines.size(); ++i) {
            cv::Point pos(topLeft.x + padding, topLeft.y + padding + (i + 1) * lineSpacing - 4);
            cv::putText(img, lines[i], pos, font, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Funciones auxiliares para SGBM
// ─────────────────────────────────────────────────────────────────────────────
static float getMedianDisparity(const cv::Mat& disp16S, cv::Rect roi) {
    roi &= cv::Rect(0, 0, disp16S.cols, disp16S.rows);
    cv::Mat crop = disp16S(roi);

    std::vector<float> vals;
    vals.reserve(crop.total());
    for (int y = 0; y < crop.rows; ++y)
        for (int x = 0; x < crop.cols; ++x)
            if (short d = crop.at<short>(y, x); d > 0)
                vals.push_back(d / 16.0f);

    if (vals.empty()) return -1.0f;
    std::sort(vals.begin(), vals.end());
    return vals[vals.size() / 2];
}

static cv::Mat buildDashboardTile(const cv::Mat& img, int w, int h) {
    if (img.empty()) return cv::Mat::zeros(h, w, CV_8UC3);
    cv::Mat out;
    if (img.channels() == 1) cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else                     out = img.clone();
    cv::resize(out, out, cv::Size(w, h));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// runLiveDisparity — pipeline SGBM principal con panel interactivo en una sola ventana
// ─────────────────────────────────────────────────────────────────────────────
void runLiveDisparity(CameraStream& cam1, CameraStream& cam2,
                      const CalibrationConfig& config) {
    Rectifier rectifier(config);
    if (!rectifier.compute()) { 
        Log::error("DisparityMode", "Fallo al inicializar rectificacion"); 
        return; 
    }

    // ── Parametros iniciales de SGBM ─────────────────────────────────────────
    SGBMProcessor::Params params;
    params.claheClipLimit    = 1.5;
    params.preBlurSigma      = 0.5;  // blur reducido: preserva textura a distancias medias
    params.numDisparities    = 144;
    params.blockSize         = 9;
    params.uniquenessRatio   = 20;
    params.speckleWindowSize = 100;  // reducido: no borrar regiones válidas a 1 metro
    params.speckleRange      = 2;
    params.wlsLambda         = 10000.0;
    params.wlsSigma          = 1.0;
    params.temporalAlpha     = 0.2;
    SGBMProcessor sgbm(params);

    // ── Ventana del Dashboard ────────────────────────────────────────────────
    const std::string winName = "Dashboard Vision Estereo";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    // ── Estado de interaccion ────────────────────────────────────────────────
    DashboardState state;
    cv::setMouseCallback(winName, onDashboardMouse, &state);

    // ── Barras de ajuste (Trackbars) (9 controles clave en orden del pipeline) 
    int tbClaheClip  = int(params.claheClipLimit * 10.0);
    int tbPreBlur    = int(params.preBlurSigma * 10.0);
    int tbNumDisp    = params.numDisparities / 16;
    int tbBlockSize  = (params.blockSize - 3) / 2;
    int tbUniqueness = params.uniquenessRatio;
    int tbSpeckle    = params.speckleWindowSize;
    int tbWlsLambda  = int(params.wlsLambda / 500.0); // 1 a 40 (500 a 20000)
    int tbWlsSigma   = int(params.wlsSigma * 10.0);   // 5 a 25 (0.5 a 2.5)
    int tbTemporal   = int(params.temporalAlpha * 10.0); // 0 a 10

    cv::createTrackbar("Contraste",    winName, &tbClaheClip,  100);
    cv::createTrackbar("Filtro",       winName, &tbPreBlur,    30);
    cv::createTrackbar("Disparidades", winName, &tbNumDisp,    10); // hasta 160 (tb * 16)
    cv::createTrackbar("Bloque",       winName, &tbBlockSize,  11); // hasta 25 (tb * 2 + 3)
    cv::createTrackbar("Unicidad",     winName, &tbUniqueness, 50);
    cv::createTrackbar("Speckle",      winName, &tbSpeckle,    250);
    cv::createTrackbar("WLS Lambda",   winName, &tbWlsLambda,  40);
    cv::createTrackbar("WLS Sigma",    winName, &tbWlsSigma,   30);
    cv::createTrackbar("Temporal",     winName, &tbTemporal,   10);

    std::cout << "\n  Modo Disparidad en Vivo — Ajuste de parametros.\n";
    std::cout << "  * HAGA CLIC en cualquier panel para ampliar/minimizar.\n";
    std::cout << "  * ESC = Volver al Menu.\n\n";

    // Kalman1D(ruidoDeProceso, ruidoDeMedida)
    Kalman1D kalmanFijo(0.01f, 9.0f);       // Q pequeño → suaviza; R=9 → σ=3cm
    Kalman1D kalmanDinamico(0.1f, 9.0f);    // Q más alto → sigue movimiento

    // Constantes físicas
    const float FOCAL_PX   = 614.76f;   
    const float BASELINE_M = 0.07531f;

    SGBMProcessor::Params lastParams = params;
    
    // Ángulo de barrido del radar
    double radarAngle = 0.0;

    int frameCount = 0;
    auto timeStart = cv::getTickCount();
    float currentFps = 0.0f;

    while (true) {
        // ── Leer barras de ajuste → actualizar parámetros ─────────────────────
        params.claheClipLimit    = std::max(0.1, tbClaheClip / 10.0);
        params.preBlurSigma      = tbPreBlur / 10.0;
        params.numDisparities    = std::max(16, tbNumDisp * 16);
        params.blockSize         = tbBlockSize * 2 + 3;
        params.uniquenessRatio   = tbUniqueness;
        params.speckleWindowSize = tbSpeckle;
        params.wlsLambda         = std::max(1.0, tbWlsLambda * 500.0);
        params.wlsSigma          = std::max(0.1, tbWlsSigma / 10.0);
        params.temporalAlpha     = std::clamp(tbTemporal / 10.0, 0.0, 1.0);

        if (params.claheClipLimit != lastParams.claheClipLimit   ||
            params.preBlurSigma   != lastParams.preBlurSigma     ||
            params.numDisparities != lastParams.numDisparities   ||
            params.blockSize      != lastParams.blockSize        ||
            params.uniquenessRatio!= lastParams.uniquenessRatio  ||
            params.speckleWindowSize != lastParams.speckleWindowSize ||
            params.wlsLambda      != lastParams.wlsLambda        ||
            params.wlsSigma       != lastParams.wlsSigma         ||
            params.temporalAlpha  != lastParams.temporalAlpha) {
            sgbm.setParams(params);
            lastParams = params;
            Log::info("DisparityMode", "Parametros SGBM actualizados");
        }

        // ── Capturar frames ───────────────────────────────────────────────────
        cv::Mat f1 = grabFrame(cam1);
        cv::Mat f2 = grabFrame(cam2);
        if (f1.empty() || f2.empty()) { cv::waitKey(1); continue; }

        frameCount++;
        if (frameCount % 10 == 0) {
            auto timeNow = cv::getTickCount();
            currentFps = 10.0f / ((timeNow - timeStart) / cv::getTickFrequency());
            timeStart = timeNow;
        }

        // ── Pipeline SGBM ─────────────────────────────────────────────────────
        auto [rL, rR]        = rectifier.rectify(f1, f2);
        cv::Mat rawDisp      = sgbm.compute(rL, rR);
        cv::Mat rawVis       = sgbm.visualize(rawDisp);
        cv::Mat claheView    = sgbm.getEnhancedLeft();
        cv::Mat wlsDisp      = sgbm.postprocess(rawDisp, rL);
        cv::Mat wlsVis       = sgbm.visualize(wlsDisp);
        cv::Mat temporalDisp = sgbm.temporalSmooth(wlsDisp);
        cv::Mat temporalVis;
        if (!temporalDisp.empty()) {
            temporalVis = sgbm.visualize(temporalDisp);
        }

        // ── HUD del tracker dinamico de AR ─────────────────────────────
        cv::Mat arView = rL.clone();
        cv::Point targetPoint = {arView.cols / 2, arView.rows / 2};
        bool targetFound      = false;
        float zDyn            = -1.0f;
        {
            double minD, maxD;
            cv::minMaxLoc(temporalDisp, &minD, &maxD);

            std::vector<std::vector<cv::Point>> contours;
            int largestIdx = -1;

            // ── Constante física: solo detectar objetos a menos de 150 cm ────────────
            // d_threshold_16S = f * B / Z_max * 16
            const int DISP_MIN_FOR_DETECTION = (int)((FOCAL_PX * BASELINE_M / 1.50) * 16.0); // ≈ 503 para 150 cm

            // Si ningún píxel supera el umbral absoluto → no hay objeto cercano
            if (maxD > DISP_MIN_FOR_DETECTION) {
                // Umbral relativo ampliado: captura toda la silueta del objeto, no sólo su punta
                double finalThresh = std::max((double)DISP_MIN_FOR_DETECTION, maxD - 160.0);
                cv::Mat mask = (temporalDisp > (int)finalThresh);
                mask.convertTo(mask, CV_8U, 1);
                
                // Kernel más grande para cerrar bien los huecos del mapa de disparidad
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15});
                cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
                cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                double maxArea = 0, minAreaTh = 200, maxAreaTh = arView.total() * 0.55;
                for (size_t i = 0; i < contours.size(); ++i) {
                    double a = cv::contourArea(contours[i]);
                    if (a > maxArea && a > minAreaTh && a < maxAreaTh) { 
                        maxArea = a; 
                        largestIdx = (int)i; 
                    }
                }
                if (largestIdx >= 0) {
                    targetFound = true;
                    cv::Moments m = cv::moments(contours[largestIdx]);
                    targetPoint  = {int(m.m10 / m.m00), int(m.m01 / m.m00)};
                }
            }

            if (targetFound) {
                cv::Rect tr = {targetPoint.x - 15, targetPoint.y - 15, 30, 30};
                float d = getMedianDisparity(rawDisp, tr);
                zDyn = kalmanDinamico.update(d > 0.0f ? (FOCAL_PX * BASELINE_M / d) * 100.0f : -1.0f);
            } else {
                kalmanDinamico.update(-1.0f);
            }

            bool validZ = (targetFound && zDyn > 0.0f && zDyn < 250.0f);
            if (validZ) {
                // t=0 → rojo (30cm)  t=1 → verde (200cm)
                // A 1 metro → t≈0.4 → naranja/cálido (sigue viéndose como "cerca")
                float t   = std::clamp((zDyn - 30.0f) / 170.0f, 0.0f, 1.0f);
                cv::Scalar col(0, int(255*t), int(255*(1-t)));

                cv::Rect bbox = cv::boundingRect(contours[largestIdx]);

                // ── EFECTO 1: Relleno semi-transparente (vision termica) ────────────
                // Dibuja el interior del contorno con el color de profundidad al 25%
                // → Efecto de camara termica / escaner sobre el objeto real
                {
                    cv::Mat overlay = arView.clone();
                    cv::drawContours(overlay, contours, largestIdx, col, cv::FILLED);
                    cv::addWeighted(arView, 0.72, overlay, 0.28, 0, arView);
                }

                // ── EFECTO 2: Contorno exterior del objeto ─────────────────────────
                cv::drawContours(arView, contours, largestIdx, col, 2, cv::LINE_AA);

                // ── EFECTO 3: Brackets que escalan con la distancia ────────────────
                // Cerca (30cm)  → L grande (50px): brackets "cierran" sobre el objetivo
                // Lejos (150cm) → L pequeño (12px): brackets pequeños y discretos
                int L = (int)std::clamp(50.0f - (zDyn - 30.0f) * 0.27f, 12.0f, 50.0f);
                int thickness = (zDyn < 60.0f) ? 3 : 2; // más gruesos si está cerca

                cv::line(arView, {bbox.x,            bbox.y},            {bbox.x+L,            bbox.y},            col, thickness);
                cv::line(arView, {bbox.x,            bbox.y},            {bbox.x,              bbox.y+L},          col, thickness);
                cv::line(arView, {bbox.x+bbox.width, bbox.y},            {bbox.x+bbox.width-L, bbox.y},            col, thickness);
                cv::line(arView, {bbox.x+bbox.width, bbox.y},            {bbox.x+bbox.width,   bbox.y+L},          col, thickness);
                cv::line(arView, {bbox.x,            bbox.y+bbox.height},{bbox.x+L,            bbox.y+bbox.height},col, thickness);
                cv::line(arView, {bbox.x,            bbox.y+bbox.height},{bbox.x,              bbox.y+bbox.height-L},col, thickness);
                cv::line(arView, {bbox.x+bbox.width, bbox.y+bbox.height},{bbox.x+bbox.width-L, bbox.y+bbox.height},col, thickness);
                cv::line(arView, {bbox.x+bbox.width, bbox.y+bbox.height},{bbox.x+bbox.width,   bbox.y+bbox.height-L},col, thickness);

                // ── EFECTO 4: Circulo pulsante en el centroide ─────────────────────
                // La velocidad del pulso aumenta cuanto mas cerca esta el objeto
                // Lejos = late lento. Cerca peligroso = late rapido (alarma)
                static double pulsePhase = 0.0;
                float proximity = std::clamp(1.0f - t, 0.0f, 1.0f); // 0=lejos, 1=cerca
                pulsePhase += 0.05 + proximity * 0.25;               // velocidad dinamica
                int pulseRadius = int(8 + 10 * std::abs(std::sin(pulsePhase)));
                int pulseAlpha  = int(120 + 135 * std::abs(std::sin(pulsePhase)));

                // Punto central solido
                cv::circle(arView, targetPoint, 4, col, -1, cv::LINE_AA);
                // Anillo pulsante exterior
                cv::Scalar pulseCol(0, int(255*t * pulseAlpha/255.0), int(255*(1-t) * pulseAlpha/255.0));
                cv::circle(arView, targetPoint, pulseRadius, col, 2, cv::LINE_AA);
                // Segundo anillo mas tenue (doble onda)
                int pulse2R = int(pulseRadius * 1.8);
                if (pulse2R < std::min(bbox.width, bbox.height) / 2)
                    cv::circle(arView, targetPoint, pulse2R, col, 1, cv::LINE_AA);

                // ── HUD texto ─────────────────────────────────────────────────────
                std::vector<std::string> info = {"Objeto detectado"};
                drawHUDTextBox(arView, info, {bbox.x, std::max(35, bbox.y - 45)}, col);

            } else {
                // Animación de barrido de radar
                cv::Point sc  = {arView.cols/2, arView.rows/2};
                cv::Scalar radarColor(120, 120, 120);
                
                cv::circle(arView, sc, 50, radarColor, 1, cv::LINE_AA);
                cv::circle(arView, sc, 25, radarColor, 1, cv::LINE_AA);
                cv::circle(arView, sc, 2, radarColor, -1, cv::LINE_AA);
                
                static double radarAngleAnim = 0.0;
                radarAngleAnim += 0.15;
                cv::Point radarEdge = {
                    sc.x + int(50 * std::cos(radarAngleAnim)),
                    sc.y + int(50 * std::sin(radarAngleAnim))
                };
                cv::line(arView, sc, radarEdge, radarColor, 1, cv::LINE_AA);
                
                std::vector<std::string> info = {"Estado: Buscando..."};
                drawHUDTextBox(arView, info, {15, 45}, radarColor);
            }
        }

        // ── Panel 05: Medicion de Profundidad Fija (Centroide) ─────────
        cv::Mat fixedDepthView = rL.clone();
        {
            cv::Point measurePoint = {fixedDepthView.cols / 2, fixedDepthView.rows / 2};

            cv::Rect roiRect = {
                std::clamp(measurePoint.x - 3, 0, temporalDisp.cols - 7),
                std::clamp(measurePoint.y - 3, 0, temporalDisp.rows - 7),
                7, 7
            };

            float disp = getMedianDisparity(temporalDisp, roiRect); 
            float z    = -1.0f;

            if (disp > 0.0f) {
                z = kalmanFijo.update((FOCAL_PX * BASELINE_M / disp) * 100.0f);
            } else {
                kalmanFijo.update(-1.0f);
            }

            bool validZ = (z > 0.0f && z < 400.0f);
            cv::Scalar color = validZ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255); // verde / rojo

            // Dibujar punto central SOLAMENTE
            cv::circle(fixedDepthView, measurePoint, 4, color, -1, cv::LINE_AA);

            // Caja de texto
            std::vector<std::string> lines;
            if (validZ) {
                std::ostringstream sz, sd, sfps;
                sz << std::fixed << std::setprecision(1) << "Z: " << z << " cm";
                sd << std::fixed << std::setprecision(1) << "Disp: " << disp << " px";
                sfps << std::fixed << std::setprecision(1) << "FPS: " << currentFps;
                lines.push_back(sz.str());
                lines.push_back(sd.str());
                lines.push_back(sfps.str());
            } else {
                lines.push_back("Z: Fuera de rango");
                std::ostringstream sfps;
                sfps << std::fixed << std::setprecision(1) << "FPS: " << currentFps;
                lines.push_back(sfps.str());
            }
            drawHUDTextBox(fixedDepthView, lines, {15, 45}, color);
        }

        // ── Renderizar las salidas de la pantalla (Ordenadas precisamente por flujo) ─
        cv::Mat fullT0 = claheView;
        cv::Mat fullT1 = rawVis;
        cv::Mat fullT2 = wlsVis;
        cv::Mat fullT3 = temporalVis.empty() ? cv::Mat::zeros(rL.size(), CV_8UC3) : temporalVis.clone();
        if (!temporalVis.empty()) {
            std::ostringstream lStr, sStr, tStr;
            lStr << std::fixed << std::setprecision(0) << params.wlsLambda;
            sStr << std::fixed << std::setprecision(1) << params.wlsSigma;
            tStr << std::fixed << std::setprecision(1) << params.temporalAlpha;
            
            std::vector<std::string> paramInfo = {
                "Parametros SGBM:",
                "Disparidad: " + std::to_string(params.numDisparities) + " px",
                "Bloque: "     + std::to_string(params.blockSize) + "x" + std::to_string(params.blockSize),
                "Unicidad: "   + std::to_string(params.uniquenessRatio),
                "Speckle: "    + std::to_string(params.speckleWindowSize),
                "WLS Lambda: " + lStr.str(),
                "WLS Sigma: "  + sStr.str(),
                "Temporal: "   + tStr.str()
            };
            drawHUDTextBox(fullT3, paramInfo, {15, fullT3.rows - 165}, cv::Scalar(180, 255, 50));
        }
        cv::Mat fullT4 = fixedDepthView;
        cv::Mat fullT5 = arView;

        // Colores
        cv::Scalar c0(255, 255, 0), c1(0, 165, 255), c2(255, 0, 255);
        cv::Scalar c3(180, 255, 50), c4(0, 255, 0), c5(255, 191, 0);

        if (state.activeTile == -1) {
            // ── Vista de cuadricula (6 paneles) ──
            cv::Mat t0 = buildDashboardTile(fullT0, 320, 240); drawTileHeader(t0, "01 - Preprocesamiento CLAHE", c0);
            cv::Mat t1 = buildDashboardTile(fullT1, 320, 240); drawTileHeader(t1, "02 - Disparidad Raw (SGBM)", c1);
            cv::Mat t2 = buildDashboardTile(fullT2, 320, 240); drawTileHeader(t2, "03 - Filtrado WLS", c2);
            cv::Mat t3 = buildDashboardTile(fullT3, 320, 240); drawTileHeader(t3, "04 - Filtro Temporal (Final)", c3);
            cv::Mat t4 = buildDashboardTile(fullT4, 320, 240); drawTileHeader(t4, "05 - Medicion de Profundidad (Mira)", c4);
            cv::Mat t5 = buildDashboardTile(fullT5, 320, 240); drawTileHeader(t5, "06 - Segmentacion y Tracking AR", c5);

            cv::Mat topRow, bottomRow, dashboard;
            cv::hconcat(std::vector<cv::Mat>{t0, t1, t2}, topRow);
            cv::hconcat(std::vector<cv::Mat>{t3, t4, t5}, bottomRow);
            cv::vconcat(topRow, bottomRow, dashboard);
            
            cv::imshow(winName, dashboard);
        } else {
            // ── Vista expandida (1 panel) ──
            cv::Mat expandedSource;
            std::string labelStr;
            cv::Scalar labelColor;

            switch (state.activeTile) {
                case 0: expandedSource = fullT0; labelStr = "01 - Preprocesamiento CLAHE"; labelColor = c0; break;
                case 1: expandedSource = fullT1; labelStr = "02 - Disparidad Raw (SGBM)"; labelColor = c1; break;
                case 2: expandedSource = fullT2; labelStr = "03 - Filtrado WLS"; labelColor = c2; break;
                case 3: expandedSource = fullT3; labelStr = "04 - Filtro Temporal (Final)"; labelColor = c3; break;
                case 4: expandedSource = fullT4; labelStr = "05 - Medicion de Profundidad (Mira)"; labelColor = c4; break;
                case 5: expandedSource = fullT5; labelStr = "06 - Segmentacion y Tracking AR"; labelColor = c5; break;
                default: expandedSource = fullT0; labelStr = "VISTA DETALLADA"; labelColor = c0; break;
            }

            cv::Mat expanded;
            cv::resize(expandedSource, expanded, cv::Size(960, 720)); // ampliar de manera nítida a 4:3
            
            // Dibujar barra superior
            cv::Rect headerStrip(0, 0, expanded.cols, 35);
            cv::Mat headerOverlay = expanded.clone();
            cv::rectangle(headerOverlay, headerStrip, cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(headerOverlay, 0.75, expanded, 0.25, 0, expanded);
            cv::line(expanded, {0, 0}, {0, 34}, labelColor, 5);
            cv::line(expanded, {0, 34}, {expanded.cols, 34}, cv::Scalar(60, 60, 60), 1);
            cv::putText(expanded, labelStr, {15, 23}, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            // Dibujar barra inferior con instrucciones para regresar
            cv::Rect footerStrip(0, expanded.rows - 30, expanded.cols, 30);
            cv::Mat footerOverlay = expanded.clone();
            cv::rectangle(footerOverlay, footerStrip, cv::Scalar(0, 0, 40), -1); 
            cv::addWeighted(footerOverlay, 0.75, expanded, 0.25, 0, expanded);
            cv::line(expanded, {0, expanded.rows - 30}, {expanded.cols, expanded.rows - 30}, cv::Scalar(40, 40, 100), 1);
            cv::putText(expanded, "HAGA CLIC PARA VOLVER A LA CUADRICULA", 
                        {20, expanded.rows - 10}, cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            cv::imshow(winName, expanded);
        }

        if (cv::waitKey(1) == 27) break;
    }
    cv::destroyWindow(winName);
}
