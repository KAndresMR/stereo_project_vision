#include <opencv2/opencv.hpp>
#include "CameraStream.hpp"
#include "CalibrationConfig.hpp"
#include "StreamWorker.hpp"
#include "CalibrationMode.hpp"
#include "MonoCalibrator.hpp"
#include "StereoCalibrator.hpp"
#include "Rectifier.hpp"
#include "SGBMProcessor.hpp"
#include "Kalman1D.hpp"
#include "DiagnosticLogger.hpp"
#include "DebugViews.hpp"
#include <thread>
#include <iostream>
#include <string>
#include <filesystem>
#include <iomanip>   
#include <algorithm> 

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// PipelineStatus
// ─────────────────────────────────────────────────────────────────────────────
struct PipelineStatus
{

    int datasetLeft = 0;
    int datasetRight = 0;

    bool leftYaml = false;
    bool rightYaml = false;
    bool stereoYaml = false;

    static PipelineStatus check(const CalibrationConfig &cfg)
    {

        PipelineStatus s;

        auto countJpg = [](const std::string &dir) -> int
        {
            if (!fs::exists(dir))
                return 0;

            int n = 0;

            for (auto &e : fs::directory_iterator(dir))
            {
                if (e.path().extension() == ".jpg")
                    n++;
            }

            return n;
        };
        s.datasetLeft = countJpg(cfg.datasetDir + "/left");
        s.datasetRight = countJpg(cfg.datasetDir + "/right");
        s.leftYaml = fs::exists(cfg.leftYaml);
        s.rightYaml = fs::exists(cfg.rightYaml);
        s.stereoYaml = fs::exists(cfg.stereoYaml);
        return s;
    }

    bool datasetReady() const
    {
        return datasetLeft >= 10 &&
                datasetRight >= 10 &&
                datasetLeft == datasetRight;
    }

    bool monoReady() const
    {
        return leftYaml && rightYaml;
    }

    bool stereoReady() const
    {
        return stereoYaml;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// printMenu
// ─────────────────────────────────────────────────────────────────────────────
static void printMenu(const PipelineStatus &st)
{
    auto icon = [](bool ok) -> std::string
    {
        return ok ? "[✓]" : "[ ]";
    };
    auto datasetIcon = [&]() -> std::string
    {
        if (st.datasetLeft == 0)
            return "[ ]";

        if (st.datasetLeft == st.datasetRight &&
            st.datasetLeft >= 10)
            return "[✓]";

        return "[~]";
    };

    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════╗\n";
    std::cout << "  ║        Stereo Vision — Pipeline Menu         ║\n";
    std::cout << "  ╚══════════════════════════════════════════════╝\n\n";

    std::cout << "  Pipeline status:\n";

    std::cout << "  " << datasetIcon()
                << " Dataset "
                << "LEFT=" << st.datasetLeft
                << " RIGHT=" << st.datasetRight
                << "\n";

    std::cout << "  " << icon(st.leftYaml)
                << " LEFT YAML\n";

    std::cout << "  " << icon(st.rightYaml)
                << " RIGHT YAML\n";

    std::cout << "  " << icon(st.stereoYaml)
                << " Stereo YAML\n";

    std::cout << "\n";

    std::cout << "  ┌─────────────────────────────────────────────┐\n";
    std::cout << "  │  Requires cameras                           │\n";
    std::cout << "  │                                             │\n";
    std::cout << "  │   1. Preview                                │\n";
    std::cout << "  │   2. Capture dataset                        │\n";
    std::cout << "  │   5. Live rectified preview                 │\n";
    std::cout << "  │   7. Live disparity (SGBM)                  │\n";
    std::cout << "  │                                             │\n";
    std::cout << "  │  Offline processing                         │\n";
    std::cout << "  │                                             │\n";
    std::cout << "  │   3. Mono calibration                       │\n";
    std::cout << "  │   4. Stereo calibration                     │\n";
    std::cout << "  │   6. Epipolar dataset check                 │\n";
    std::cout << "  │                                             │\n";
    std::cout << "  │   0. Exit                                   │\n";
    std::cout << "  └─────────────────────────────────────────────┘\n";

    std::cout << "\n  Choice: ";
}

// ─────────────────────────────────────────────────────────────────────────────
// waitForCameras
// ─────────────────────────────────────────────────────────────────────────────
static void waitForCameras(
    CameraStream &cam1,
    CameraStream &cam2)
{
    std::cout << "\n  Connecting to cameras";
    std::cout.flush();
    while (true)
    {
        bool ok1;
        bool ok2;
        {
            std::lock_guard<std::mutex> l(cam1.frame_mtx);
            ok1 = !cam1.frame.empty();
        }
        {
            std::lock_guard<std::mutex> l(cam2.frame_mtx);
            ok2 = !cam2.frame.empty();
        }
        if (ok1 && ok2)
            break;
        std::cout << ".";
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << " OK\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// runPreviewMode
// ─────────────────────────────────────────────────────────────────────────────
static void runPreviewMode(
    CameraStream &cam1,
    CameraStream &cam2)
{
    while (true)
    {

        cv::Mat f1;
        cv::Mat f2;

        {
            std::lock_guard<std::mutex> l(cam1.frame_mtx);

            if (!cam1.frame.empty())
                f1 = cam1.frame.clone();
        }

        {
            std::lock_guard<std::mutex> l(cam2.frame_mtx);

            if (!cam2.frame.empty())
                f2 = cam2.frame.clone();
        }

        if (!f1.empty())
            cv::imshow("LEFT", f1);

        if (!f2.empty())
            cv::imshow("RIGHT", f2);

        int key = cv::waitKey(1);

        if (key == 27)
            break;
    }

    cv::destroyAllWindows();
}

// ─────────────────────────────────────────────────────────────────────────────
// runMonoCalibration
// ─────────────────────────────────────────────────────────────────────────────
static void runMonoCalibration(
    const CalibrationConfig &config)
{
    Log::separator("PHASE 1 — Mono Calibration");

    MonoCalibrator mc(config);

    auto left = mc.calibrate(MonoCalibrator::Side::LEFT);

    if (left.success)
        mc.saveYAML(left,
                    MonoCalibrator::Side::LEFT);

    auto right = mc.calibrate(MonoCalibrator::Side::RIGHT);

    if (right.success)
        mc.saveYAML(right,
                    MonoCalibrator::Side::RIGHT);
}

// ─────────────────────────────────────────────────────────────────────────────
// runStereoCalibration
// ─────────────────────────────────────────────────────────────────────────────
static void runStereoCalibration(
    const CalibrationConfig &config)
{
    Log::separator("PHASE 2 — Stereo Calibration");

    StereoCalibrator stereo(config);

    auto result = stereo.calibrate();

    if (!result.success)
    {

        Log::error(
            "Main",
            "Stereo calibration failed");

        return;
    }

    stereo.saveYAML(result);

    Log::info(
        "Main",
        "Stereo YAML saved");
}

// ─────────────────────────────────────────────────────────────────────────────
// runEpipolarDatasetCheck
// ─────────────────────────────────────────────────────────────────────────────
static void runEpipolarDatasetCheck(
    const CalibrationConfig &config)
{
    Rectifier rect(config);

    if (!rect.compute())
    {

        Log::error(
            "Main",
            "Rectification failed");

        return;
    }

    rect.previewDataset();
}

// ─────────────────────────────────────────────────────────────────────────────
// runLiveRectifiedPreview
// ─────────────────────────────────────────────────────────────────────────────
static void runLiveRectifiedPreview(
    CameraStream &cam1,
    CameraStream &cam2,
    const CalibrationConfig &config)
{
    Rectifier rect(config);

    if (!rect.compute())
    {

        Log::error(
            "Main",
            "Rectification init failed");

        return;
    }

    while (true)
    {

        cv::Mat f1;
        cv::Mat f2;

        {
            std::lock_guard<std::mutex> l(cam1.frame_mtx);

            if (!cam1.frame.empty())
                f1 = cam1.frame.clone();
        }

        {
            std::lock_guard<std::mutex> l(cam2.frame_mtx);

            if (!cam2.frame.empty())
                f2 = cam2.frame.clone();
        }

        if (!f1.empty() && !f2.empty())
        {

            auto [rL, rR] =
                rect.rectify(f1, f2);

            cv::Mat debug =
                rect.drawEpipolarLines(rL, rR);

            cv::imshow(
                "Live Rectified",
                debug);
        }

        int key = cv::waitKey(1);

        if (key == 27)
            break;
    }

    
}

// Función Helper: Extraer la mediana de la disparidad en un ROI (Región de Interés)
float getMedianDisparity(const cv::Mat& disp16S, cv::Rect roi) {
    // Asegurarnos de que el ROI no se salga de la imagen
    roi &= cv::Rect(0, 0, disp16S.cols, disp16S.rows);
    cv::Mat crop = disp16S(roi);

    std::vector<float> validDisparities;
    validDisparities.reserve(crop.total());

    for (int y = 0; y < crop.rows; y++) {
        for (int x = 0; x < crop.cols; x++) {
            short d = crop.at<short>(y, x);
            if (d > 0) { // Descartar píxeles inválidos
                validDisparities.push_back(d / 16.0f); // CV_16S está multiplicado por 16
            }
        }
    }

    if (validDisparities.empty()) return -1.0f; // No hay datos válidos en el centro

    // Encontrar la mediana matemáticamente
    std::sort(validDisparities.begin(), validDisparities.end());
    return validDisparities[validDisparities.size() / 2];
}

// ─────────────────────────────────────────────────────────────────────────────
// runLiveDisparity
// ─────────────────────────────────────────────────────────────────────────────
static void runLiveDisparity(
    CameraStream &cam1,
    CameraStream &cam2,
    const CalibrationConfig &config)
{
    Rectifier rectifier(config);

    if (!rectifier.compute())
    {

        Log::error(
            "Main",
            "Rectification init failed");

        return;
    }

    // ── Initial SGBM params ──────────────────────────────────────────────
    SGBMProcessor::Params params;

    params.numDisparities = 64;
    params.blockSize = 7;
    params.uniquenessRatio = 10;
    params.speckleWindowSize = 100;
    params.speckleRange = 32;

    SGBMProcessor sgbm(params);

    // ── Trackbar state y Ventana Única ───────────────────────────────────
    std::string winName = "Stereo Vision Dashboard";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    int tbNumDisp = 6; // 4*16 = 64
    int tbBlockSize = 7;
    int tbUniqueness = 15;
    int tbSpeckleWin = 100;
    int tbClaheClip = 25; // 40 / 10.0 = 4.0
    int tbClaheTile = 8;  // Cuadrículas de 8x8
    int tbPreBlur = 8;

    // Asignamos TODOS los trackbars a la misma ventana "Stereo Vision Dashboard"
    cv::createTrackbar("numDisp x16", winName, &tbNumDisp, 20);
    cv::createTrackbar("blockSize",   winName, &tbBlockSize, 21);
    cv::createTrackbar("uniqueness",  winName, &tbUniqueness, 50);
    cv::createTrackbar("speckleWin",  winName, &tbSpeckleWin, 200);
    cv::createTrackbar("CLAHE Clip",  winName, &tbClaheClip, 100);
    cv::createTrackbar("CLAHE Tile",  winName, &tbClaheTile, 32);
    cv::createTrackbar("Pre-Blur",    winName, &tbPreBlur, 30);

    std::cout << "\n";
    std::cout << "  Live SGBM mode\n";
    std::cout << "  Adjust parameters with sliders\n";
    std::cout << "  ESC = back to menu\n\n";

    SGBMProcessor::Params lastParams = params;
    
    // Función helper CORREGIDA (añadimos -> cv::Mat)
    auto formatForDashboard = [](const cv::Mat& img, int width, int height) -> cv::Mat {
        cv::Mat out;
        if (img.empty()) {
            return cv::Mat::zeros(height, width, CV_8UC3); 
        }
        if (img.channels() == 1) {
            cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
        } else {
            out = img.clone();
        }
        cv::resize(out, out, cv::Size(width, height));
        return out;
    };

    Kalman1D kalmanFilter(1e-3, 0.1); 
    
    // Tus constantes físicas obtenidas de la calibración
    const float FOCAL_LENGTH_PX = 600.0f; // Promedio entre fx (600.6) y fy (599.8)
    const float BASELINE_M = 0.07414f;     // 74.14 mm a cm

    while (true)
    {
        // ── CONSTRUCCIÓN DEL DASHBOARD UNIFICADO ───────────────────

        // ── Update params from trackbars ────────────────────────────────
        params.numDisparities = std::max(16, tbNumDisp * 16);
        params.blockSize = std::max(3, tbBlockSize);

        if (params.blockSize % 2 == 0)
            params.blockSize++;

        params.uniquenessRatio = tbUniqueness;
        params.speckleWindowSize = tbSpeckleWin;
        params.claheClipLimit = std::max(0.1, tbClaheClip / 10.0);
        params.claheTileSize = std::max(2, tbClaheTile);
        params.preBlurSigma = tbPreBlur/10.0;

        bool paramsChanged =
            params.numDisparities != lastParams.numDisparities ||
            params.blockSize != lastParams.blockSize ||
            params.uniquenessRatio != lastParams.uniquenessRatio ||
            params.speckleWindowSize != lastParams.speckleWindowSize ||
            params.claheClipLimit != lastParams.claheClipLimit ||
            params.claheTileSize != lastParams.claheTileSize ||
            params.preBlurSigma != lastParams.preBlurSigma;

        if (paramsChanged)
        {
            sgbm.setParams(params);
            lastParams = params;
            Log::info(
                "Main",
                "SGBM params updated");
        }
        // ── Grab frames ─────────────────────────────────────────────────
        cv::Mat f1;
        cv::Mat f2;
        {
            std::lock_guard<std::mutex> l(cam1.frame_mtx);
            if (!cam1.frame.empty())
                f1 = cam1.frame.clone();
        }
        {
            std::lock_guard<std::mutex> l(cam2.frame_mtx);
            if (!cam2.frame.empty())
                f2 = cam2.frame.clone();
        }
        if (!f1.empty() && !f2.empty())
        {
            // 1. RECTIFICACIÓN Y PIPELINE SGBM
            auto [rL, rR] = rectifier.rectify(f1, f2);

            cv::Mat rawDisp = sgbm.compute(rL, rR);
            cv::Mat rawVis = sgbm.visualize(rawDisp);
            cv::Mat claheView = sgbm.getEnhancedLeft();
            cv::Mat wlsDisp = sgbm.postprocess(rawDisp, rL);
            cv::Mat wlsVis = sgbm.visualize(wlsDisp);
            cv::Mat temporalDisp = sgbm.temporalSmooth(wlsDisp);
            cv::Mat temporalVis;
            
            if (!temporalDisp.empty()) {
                temporalVis = sgbm.visualize(temporalDisp);
                cv::putText(temporalVis, 
                    "numDisp=" + std::to_string(params.numDisparities) + " block=" + std::to_string(params.blockSize),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            }

            // ─────────────────────────────────────────────────────────────────
            // 2. VENTANA 1: CÁLCULO Z ESTABILIZADO (KALMAN)
            // ─────────────────────────────────────────────────────────────────
            
            cv::Mat kalmanView = rL.clone();
            
            int roiSize = 20;
            cv::Rect centerRoi((temporalDisp.cols - roiSize)/2, (temporalDisp.rows - roiSize)/2, roiSize, roiSize);
            float medianDisp = getMedianDisparity(temporalDisp, centerRoi);
            
            // [!] LA SOLUCIÓN: Declaramos la variable AFUERA para que viva en todo el bloque
            float stableDepthCm = -1.0f; 

            if (medianDisp > 0.0f) {
                // Asegúrate de usar tu constante de baseline (ej. BASELINE_CM = 7.414f) 
                // Si tienes BASELINE_M (0.07414f), multiplica al final por 100.0f
                float rawDepthCm = ((FOCAL_LENGTH_PX * BASELINE_M) / medianDisp) * 100.0f;
                
                // Asignamos el valor a la variable que ya creamos afuera
                stableDepthCm = kalmanFilter.update(rawDepthCm);

                if (stableDepthCm > 0.0f && stableDepthCm < 400.0f) {
                    cv::Point centerPoint(temporalDisp.cols / 2, temporalDisp.rows / 2);
                    cv::circle(kalmanView, centerPoint, 5, cv::Scalar(0, 255, 0), -1);

                    std::ostringstream kalmanStr;
                    kalmanStr << std::fixed << std::setprecision(2) << "Z: " << stableDepthCm << " cm";
                    cv::putText(kalmanView, kalmanStr.str(), cv::Point(centerPoint.x + 15, centerPoint.y), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                }
            }
            
            cv::imshow("1 - Medicion de Profundidad", kalmanView);

            // ─────────────────────────────────────────────────────────────────
            // 3. VENTANA 2: EFECTO DE REALIDAD AUMENTADA (TARGETING HUD)
            // ─────────────────────────────────────────────────────────────────
            
            // Clonamos la cámara izquierda para dibujar encima sin ensuciarla
            cv::Mat arFrame = rL.clone();
            cv::Point centerPoint(arFrame.cols / 2, arFrame.rows / 2);

            // Ahora sí reconocerá stableDepthCm sin problemas
            if (stableDepthCm > 0.0f && stableDepthCm < 400.0f) {
                // 1. Matemáticas de Reactividad (Mapear Z a tamaño y color)
                // Hacemos que 't' vaya de 0 (cerca, 30cm) a 1 (lejos, 150cm)
                float t = std::clamp((stableDepthCm - 30.0f) / 120.0f, 0.0f, 1.0f);

                // 2. Color dinámico: Rojo (cerca) -> Amarillo -> Verde (lejos)
                int r = (int)(255 * (1.0f - t)); // Más cerca = Más rojo
                int g = (int)(255 * t);          // Más lejos = Más verde
                cv::Scalar hudColor(0, g, r);    // Formato BGR de OpenCV

                // 3. Tamaño dinámico: La mira se hace pequeña (precisa) al acercarse
                int radius = (int)(40 + 60 * t); // Va de 40px (cerca) a 100px (lejos)

                // 4. Dibujar la interfaz (HUD)
                cv::circle(arFrame, centerPoint, radius, hudColor, 2, cv::LINE_AA);
                cv::circle(arFrame, centerPoint, radius + 10, hudColor, 1, cv::LINE_AA);
                cv::circle(arFrame, centerPoint, 3, hudColor, -1, cv::LINE_AA);

                // Aspas de la mira
                int lineLen = 15;
                cv::line(arFrame, cv::Point(centerPoint.x - radius, centerPoint.y), 
                         cv::Point(centerPoint.x - radius + lineLen, centerPoint.y), hudColor, 2);
                cv::line(arFrame, cv::Point(centerPoint.x + radius, centerPoint.y), 
                         cv::Point(centerPoint.x + radius - lineLen, centerPoint.y), hudColor, 2);
                cv::line(arFrame, cv::Point(centerPoint.x, centerPoint.y - radius), 
                         cv::Point(centerPoint.x, centerPoint.y - radius + lineLen), hudColor, 2);
                cv::line(arFrame, cv::Point(centerPoint.x, centerPoint.y + radius), 
                         cv::Point(centerPoint.x, centerPoint.y + radius - lineLen), hudColor, 2);

                // Texto dinámico
                std::ostringstream distStr;
                distStr << std::fixed << std::setprecision(1) << "TARGET LOCKED: " << stableDepthCm << " cm";
                cv::putText(arFrame, distStr.str(), cv::Point(centerPoint.x - 90, centerPoint.y - radius - 20),
                            cv::FONT_HERSHEY_DUPLEX, 0.5, hudColor, 1, cv::LINE_AA);
            } 
            else {
                // MODO BÚSQUEDA
                cv::Scalar searchColor(150, 150, 150); // Gris
                cv::circle(arFrame, centerPoint, 60, searchColor, 1, cv::LINE_AA);
                
                static double angle = 0.0;
                angle += 0.1;
                int dx = (int)(60 * std::cos(angle));
                int dy = (int)(60 * std::sin(angle));
                cv::line(arFrame, centerPoint, cv::Point(centerPoint.x + dx, centerPoint.y + dy), searchColor, 1);

                cv::putText(arFrame, "SEARCHING...", cv::Point(centerPoint.x - 50, centerPoint.y - 75),
                            cv::FONT_HERSHEY_DUPLEX, 0.5, searchColor, 1, cv::LINE_AA);
            }

            cv::imshow("2 - Efecto Realidad Aumentada", arFrame);

            // 3. CONSTRUCCIÓN DE LA VISTA ESTÉREO (Una sola vez)
            cv::Mat stereoView;
            cv::hconcat(rL, rR, stereoView);
            for (int y = 0; y < stereoView.rows; y += 40) {
                cv::line(stereoView, cv::Point(0, y), cv::Point(stereoView.cols, y), cv::Scalar(0, 255, 0), 1);
            }

            // 4. CONSTRUCCIÓN DEL DASHBOARD
            cv::Mat view1 = formatForDashboard(claheView, 320, 240);
            DebugViews::label(view1, "00 - CLAHE");
            
            cv::Mat view2 = formatForDashboard(rawVis, 320, 240);
            DebugViews::label(view2, "01 - RAW SGBM");
            
            cv::Mat view3 = formatForDashboard(wlsVis, 320, 240);
            DebugViews::label(view3, "02 - WLS FILTER");

            cv::Mat topRow;
            cv::hconcat(std::vector<cv::Mat>{view1, view2, view3}, topRow);

            cv::Mat view4 = formatForDashboard(stereoView, 640, 240);
            DebugViews::label(view4, "RECTIFIED PAIR (Epipolar Check)");
            
            cv::Mat view5 = formatForDashboard(temporalVis, 320, 240);
            DebugViews::label(view5, "03 - TEMPORAL (FINAL)");

            cv::Mat bottomRow;
            cv::hconcat(view4, view5, bottomRow);

            cv::Mat dashboard;
            cv::vconcat(topRow, bottomRow, dashboard);

            cv::imshow("Stereo Vision Dashboard", dashboard);
        }

        int key = cv::waitKey(1);

        if (key == 27)
            break;
    }

    cv::destroyWindow(winName);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{

    // ── Config ───────────────────────────────────────────────────────────
    CalibrationConfig config;

    config.boardSize = {9, 6};
    config.squareSizeM = 0.025f;

    config.targetPairs = 30;

    config.datasetDir = "calib_pairs";

    config.leftYaml =
        "calib_pairs/calib/left.yaml";

    config.rightYaml =
        "calib_pairs/calib/right.yaml";

    config.stereoYaml =
        "calib_pairs/calib/stereo.yaml";

    // ── Camera streams ──────────────────────────────────────────────────
    bool camerasRunning = false;

    CameraStream cam1;
    CameraStream cam2;

    std::thread t1;
    std::thread t2;

    auto ensureCamerasRunning = [&]()
    {
        if (camerasRunning)
            return;

        std::cout << "\n";
        std::cout << "  Starting camera streams...\n";

        t1 = std::thread(
            streamCamera,
            "http://192.168.18.111:81/stream",
            std::ref(cam1));

        t2 = std::thread(
            streamCamera,
            "http://192.168.18.112:81/stream",
            std::ref(cam2));

        waitForCameras(cam1, cam2);

        camerasRunning = true;
    };

    // ── Main menu loop ──────────────────────────────────────────────────
    while (true)
    {

        PipelineStatus st =
            PipelineStatus::check(config);

        printMenu(st);

        std::string input;

        std::getline(std::cin, input);

        input.erase(
            0,
            input.find_first_not_of(" \t"));

        if (input.empty())
            continue;

        // ────────────────────────────────────────────────────────────────
        if (input == "0")
        {

            std::cout << "\n";
            std::cout << "  Goodbye.\n\n";

            break;
        }

        // ────────────────────────────────────────────────────────────────
        if (input == "1")
        {

            ensureCamerasRunning();

            runPreviewMode(cam1, cam2);

            continue;
        }

        // ────────────────────────────────────────────────────────────────
        if (input == "2")
        {

            ensureCamerasRunning();

            runCalibrationMode(
                cam1,
                cam2,
                config);

            continue;
        }

        // ────────────────────────────────────────────────────────────────
        if (input == "3")
        {

            if (!st.datasetReady())
            {

                std::cout << "\n";
                std::cout << "  Dataset not ready.\n";

                continue;
            }

            runMonoCalibration(config);

            std::cout << "\nPress ENTER...";
            std::getline(std::cin, input);

            continue;
        }

        // ────────────────────────────────────────────────────────────────
        if (input == "4")
        {

            if (!st.monoReady())
            {

                std::cout << "\n";
                std::cout << "  Run mono calibration first.\n";

                continue;
            }

            runStereoCalibration(config);

            std::cout << "\nPress ENTER...";
            std::getline(std::cin, input);

            continue;
        }

        // ────────────────────────────────────────────────────────────────
        if (input == "5")
        {

            if (!st.stereoReady())
            {

                std::cout << "\n";
                std::cout << "  Run stereo calibration first.\n";

                continue;
            }

            ensureCamerasRunning();

            runLiveRectifiedPreview(
                cam1,
                cam2,
                config);

            continue;
        }

        // ────────────────────────────────────────────────────────────────
        if (input == "6")
        {

            if (!st.stereoReady())
            {

                std::cout << "\n";
                std::cout << "  Run stereo calibration first.\n";

                continue;
            }

            runEpipolarDatasetCheck(config);

            continue;
        }

        // ────────────────────────────────────────────────────────────────
        if (input == "7")
        {

            if (!st.stereoReady())
            {

                std::cout << "\n";
                std::cout << "  Run stereo calibration first.\n";

                continue;
            }

            ensureCamerasRunning();

            runLiveDisparity(
                cam1,
                cam2,
                config);

            continue;
        }

        // ────────────────────────────────────────────────────────────────
        std::cout << "\n";
        std::cout << "  Unknown option.\n";
    }

    // ── Shutdown ─────────────────────────────────────────────────────────
    if (camerasRunning)
    {

        running = false;

        cv::destroyAllWindows();

        t1.join();
        t2.join();
    }

    return 0;
}