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
#include <sstream>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// PipelineStatus — checks which pipeline stages are complete
// ─────────────────────────────────────────────────────────────────────────────
struct PipelineStatus {
    int  datasetLeft  = 0;
    int  datasetRight = 0;
    bool leftYaml     = false;
    bool rightYaml    = false;
    bool stereoYaml   = false;

    static PipelineStatus check(const CalibrationConfig& cfg) {
        auto countJpg = [](const std::string& dir) -> int {
            if (!fs::exists(dir)) return 0;
            int n = 0;
            for (auto& e : fs::directory_iterator(dir))
                if (e.path().extension() == ".jpg") n++;
            return n;
        };
        PipelineStatus s;
        s.datasetLeft  = countJpg(cfg.datasetDir + "/left");
        s.datasetRight = countJpg(cfg.datasetDir + "/right");
        s.leftYaml     = fs::exists(cfg.leftYaml);
        s.rightYaml    = fs::exists(cfg.rightYaml);
        s.stereoYaml   = fs::exists(cfg.stereoYaml);
        return s;
    }

    bool datasetReady() const { return datasetLeft >= 10 && datasetRight >= 10 && datasetLeft == datasetRight; }
    bool monoReady()    const { return leftYaml && rightYaml; }
    bool stereoReady()  const { return stereoYaml; }
};

// ─────────────────────────────────────────────────────────────────────────────
// printMenu
// ─────────────────────────────────────────────────────────────────────────────
static void printMenu(const PipelineStatus& st) {
    auto icon = [](bool ok) { return ok ? "[✓]" : "[ ]"; };
    auto dsIcon = [&]() -> const char* {
        if (st.datasetLeft == 0) return "[ ]";
        return (st.datasetLeft == st.datasetRight && st.datasetLeft >= 10) ? "[✓]" : "[~]";
    };

    std::cout
        << "\n"
        << "  ╔══════════════════════════════════════════════╗\n"
        << "  ║        Stereo Vision — Pipeline Menu         ║\n"
        << "  ╚══════════════════════════════════════════════╝\n\n"
        << "  Pipeline status:\n"
        << "  " << dsIcon()       << " Dataset LEFT=" << st.datasetLeft << " RIGHT=" << st.datasetRight << "\n"
        << "  " << icon(st.leftYaml)   << " LEFT YAML\n"
        << "  " << icon(st.rightYaml)  << " RIGHT YAML\n"
        << "  " << icon(st.stereoYaml) << " Stereo YAML\n\n"
        << "  ┌─────────────────────────────────────────────┐\n"
        << "  │  Requires cameras                           │\n"
        << "  │   1. Capture dataset                        │\n"
        << "  │   4. Live rectified preview                 │\n"
        << "  │   6. Live disparity (SGBM)                  │\n"
        << "  │                                             │\n"
        << "  │  Offline processing                         │\n"
        << "  │   2. Mono calibration                       │\n"
        << "  │   3. Stereo calibration                     │\n"
        << "  │   5. Epipolar dataset check                 │\n"
        << "  │                                             │\n"
        << "  │   0. Exit                                   │\n"
        << "  └─────────────────────────────────────────────┘\n"
        << "\n  Choice: ";
}

// ─────────────────────────────────────────────────────────────────────────────
// grabFrame — thread-safe frame snapshot
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat grabFrame(CameraStream& cam) {
    std::lock_guard<std::mutex> l(cam.frame_mtx);
    return cam.frame.empty() ? cv::Mat{} : cam.frame.clone();
}

// ─────────────────────────────────────────────────────────────────────────────
// waitForCameras
// ─────────────────────────────────────────────────────────────────────────────
static void waitForCameras(CameraStream& cam1, CameraStream& cam2) {
    std::cout << "\n  Connecting to cameras";
    std::cout.flush();
    while (grabFrame(cam1).empty() || grabFrame(cam2).empty()) {
        std::cout << ".";
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << " OK\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// runMonoCalibration
// ─────────────────────────────────────────────────────────────────────────────
static void runMonoCalibration(const CalibrationConfig& config) {
    Log::separator("PHASE 1 — Mono Calibration");
    MonoCalibrator mc(config);
    auto left  = mc.calibrate(MonoCalibrator::Side::LEFT);
    if (left.success)  mc.saveYAML(left,  MonoCalibrator::Side::LEFT);
    auto right = mc.calibrate(MonoCalibrator::Side::RIGHT);
    if (right.success) mc.saveYAML(right, MonoCalibrator::Side::RIGHT);
}

// ─────────────────────────────────────────────────────────────────────────────
// runStereoCalibration
// ─────────────────────────────────────────────────────────────────────────────
static void runStereoCalibration(const CalibrationConfig& config) {
    Log::separator("PHASE 2 — Stereo Calibration");
    StereoCalibrator stereo(config);
    auto result = stereo.calibrate();
    if (!result.success) { Log::error("Main", "Stereo calibration failed"); return; }
    stereo.saveYAML(result);
    Log::info("Main", "Stereo YAML saved");
}

// ─────────────────────────────────────────────────────────────────────────────
// runEpipolarDatasetCheck
// ─────────────────────────────────────────────────────────────────────────────
static void runEpipolarDatasetCheck(const CalibrationConfig& config) {
    Rectifier rect(config);
    if (!rect.compute()) { Log::error("Main", "Rectification failed"); return; }
    rect.previewDataset();
}

// ─────────────────────────────────────────────────────────────────────────────
// runLiveRectifiedPreview
// ─────────────────────────────────────────────────────────────────────────────
static void runLiveRectifiedPreview(CameraStream& cam1, CameraStream& cam2,
                                    const CalibrationConfig& config) {
    Rectifier rect(config);
    if (!rect.compute()) { Log::error("Main", "Rectification init failed"); return; }

    while (true) {
        cv::Mat f1 = grabFrame(cam1);
        cv::Mat f2 = grabFrame(cam2);
        if (!f1.empty() && !f2.empty()) {
            auto [rL, rR] = rect.rectify(f1, f2);
            cv::imshow("Live Rectified", rect.drawEpipolarLines(rL, rR));
        }
        if (cv::waitKey(1) == 27) break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// getMedianDisparity — median disparity in a ROI (CV_16S, returns px units)
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

// ─────────────────────────────────────────────────────────────────────────────
// buildDashboardTile — resize + convert to BGR for the dashboard grid
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat buildDashboardTile(const cv::Mat& img, int w, int h) {
    if (img.empty()) return cv::Mat::zeros(h, w, CV_8UC3);
    cv::Mat out;
    if (img.channels() == 1) cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else                     out = img.clone();
    cv::resize(out, out, cv::Size(w, h));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// runLiveDisparity — main SGBM pipeline with AR overlay and dashboard
// ─────────────────────────────────────────────────────────────────────────────
static void runLiveDisparity(CameraStream& cam1, CameraStream& cam2,
                             const CalibrationConfig& config) {
    Rectifier rectifier(config);
    if (!rectifier.compute()) { Log::error("Main", "Rectification init failed"); return; }

    // ── SGBM initial params ───────────────────────────────────────────────────
    SGBMProcessor::Params params;
    params.numDisparities   = 64;
    params.blockSize        = 7;
    params.uniquenessRatio  = 10;
    params.speckleWindowSize = 100;
    params.speckleRange     = 2;
    SGBMProcessor sgbm(params);

    // ── Dashboard window + trackbars ──────────────────────────────────────────
    const std::string winName = "Stereo Vision Dashboard";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    int tbNumDisp    = 8;   // *16 → 64
    int tbBlockSize  = 9;
    int tbUniqueness = 25;
    int tbSpeckleWin = 200;
    int tbClaheClip  = 25;  // /10 → 2.5
    int tbClaheTile  = 8;
    int tbPreBlur    = 8;   // /10 → 0.8

    cv::createTrackbar("numDisp x16", winName, &tbNumDisp,    20);
    cv::createTrackbar("blockSize",   winName, &tbBlockSize,  21);
    cv::createTrackbar("uniqueness",  winName, &tbUniqueness, 50);
    cv::createTrackbar("speckleWin",  winName, &tbSpeckleWin, 200);
    cv::createTrackbar("CLAHE Clip",  winName, &tbClaheClip,  100);
    cv::createTrackbar("CLAHE Tile",  winName, &tbClaheTile,  32);
    cv::createTrackbar("Pre-Blur",    winName, &tbPreBlur,    30);

    std::cout << "\n  Live SGBM mode — adjust sliders — ESC = menu\n\n";

    // ── Kalman filters (one per measurement window) ───────────────────────────
    Kalman1D kalmanFijo(1e-3f, 0.1f);      // centre cross-hair
    Kalman1D kalmanDinamico(1e-3f, 0.1f);  // AR target tracker

    // Physical constants from stereo calibration
    const float FOCAL_PX  = 600.0f;   // avg of fx=600.6 fy=599.8
    const float BASELINE_M = 0.07414f; // 74.14 mm

    SGBMProcessor::Params lastParams = params;

    while (true) {
        // ── Read trackbars → update params ────────────────────────────────────
        params.numDisparities    = std::max(16, tbNumDisp * 16);
        params.blockSize         = std::max(3, tbBlockSize | 1); // force odd
        params.uniquenessRatio   = tbUniqueness;
        params.speckleWindowSize = tbSpeckleWin;
        params.claheClipLimit    = std::max(0.1, tbClaheClip / 10.0);
        params.claheTileSize     = std::max(2, tbClaheTile);
        params.preBlurSigma      = tbPreBlur / 10.0;

        if (params.numDisparities != lastParams.numDisparities ||
            params.blockSize        != lastParams.blockSize        ||
            params.uniquenessRatio  != lastParams.uniquenessRatio  ||
            params.speckleWindowSize!= lastParams.speckleWindowSize||
            params.claheClipLimit   != lastParams.claheClipLimit   ||
            params.claheTileSize    != lastParams.claheTileSize    ||
            params.preBlurSigma     != lastParams.preBlurSigma) {
            sgbm.setParams(params);
            lastParams = params;
            Log::info("Main", "SGBM params updated");
        }

        // ── Grab frames ───────────────────────────────────────────────────────
        cv::Mat f1 = grabFrame(cam1);
        cv::Mat f2 = grabFrame(cam2);
        if (f1.empty() || f2.empty()) { cv::waitKey(1); continue; }

        // ── SGBM pipeline ─────────────────────────────────────────────────────
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
            cv::putText(temporalVis,
                "numDisp=" + std::to_string(params.numDisparities) +
                " block="  + std::to_string(params.blockSize),
                {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255,255,255}, 2);
        }

        // ── Window 1: fixed centre depth measurement ──────────────────────────
        {
            cv::Mat view      = rL.clone();
            cv::Point centre  = {view.cols / 2, view.rows / 2};
            cv::Rect  roiRect = {centre.x - 10, centre.y - 10, 20, 20};
            float disp        = getMedianDisparity(rawDisp, roiRect);
            float z           = -1.0f;
            if (disp > 0.0f) {
                z = kalmanFijo.update((FOCAL_PX * BASELINE_M / disp) * 100.0f);
                if (z > 0.0f && z < 400.0f) {
                    cv::circle(view, centre, 5, {0,255,0}, -1);
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(2) << "Z: " << z << " cm";
                    cv::putText(view, ss.str(), {centre.x + 15, centre.y},
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);
                }
            } else {
                kalmanFijo.update(-1.0f);
            }
            cv::imshow("1 - Medicion de Profundidad", view);
        }

        // ── Window 2: AR dynamic target tracker ──────────────────────────────
        {
            cv::Mat arFrame = rL.clone();
            double  minD, maxD;
            cv::minMaxLoc(temporalDisp, &minD, &maxD);

            cv::Point targetPoint = {arFrame.cols / 2, arFrame.rows / 2};
            bool targetFound      = false;
            std::vector<std::vector<cv::Point>> contours;
            int largestIdx = -1;

            if (maxD > 16.0) {
                cv::Mat mask = (temporalDisp > (maxD - 80));
                mask.convertTo(mask, CV_8U, 1);
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9,9});
                cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                double maxArea = 0, minAreaTh = 400, maxAreaTh = arFrame.total() * 0.40;
                for (size_t i = 0; i < contours.size(); ++i) {
                    double a = cv::contourArea(contours[i]);
                    if (a > maxArea && a > minAreaTh && a < maxAreaTh) { maxArea = a; largestIdx = (int)i; }
                }
                if (largestIdx >= 0) {
                    targetFound = true;
                    cv::Moments m = cv::moments(contours[largestIdx]);
                    targetPoint  = {int(m.m10 / m.m00), int(m.m01 / m.m00)};
                }
            }

            float zDyn = -1.0f;
            if (targetFound) {
                cv::Rect tr = {targetPoint.x - 10, targetPoint.y - 10, 20, 20};
                float d = getMedianDisparity(rawDisp, tr);
                zDyn = kalmanDinamico.update(d > 0.0f ? (FOCAL_PX * BASELINE_M / d) * 100.0f : -1.0f);
            } else {
                kalmanDinamico.update(-1.0f);
            }

            if (targetFound && zDyn > 0.0f && zDyn < 150.0f) {
                float t   = std::clamp((zDyn - 30.0f) / 120.0f, 0.0f, 1.0f);
                cv::Scalar col(0, int(255*t), int(255*(1-t)));
                cv::drawContours(arFrame, contours, largestIdx, col, 3, cv::LINE_AA);
                cv::Rect bbox = cv::boundingRect(contours[largestIdx]);
                const int L   = 20;
                // Sci-fi corner brackets
                cv::line(arFrame, {bbox.x,            bbox.y},            {bbox.x+L,         bbox.y},            col, 2);
                cv::line(arFrame, {bbox.x,            bbox.y},            {bbox.x,            bbox.y+L},          col, 2);
                cv::line(arFrame, {bbox.x+bbox.width, bbox.y},            {bbox.x+bbox.width-L,bbox.y},           col, 2);
                cv::line(arFrame, {bbox.x+bbox.width, bbox.y},            {bbox.x+bbox.width, bbox.y+L},          col, 2);
                cv::line(arFrame, {bbox.x,            bbox.y+bbox.height},{bbox.x+L,          bbox.y+bbox.height},col, 2);
                cv::line(arFrame, {bbox.x,            bbox.y+bbox.height},{bbox.x,            bbox.y+bbox.height-L},col,2);
                cv::line(arFrame, {bbox.x+bbox.width, bbox.y+bbox.height},{bbox.x+bbox.width-L,bbox.y+bbox.height},col,2);
                cv::line(arFrame, {bbox.x+bbox.width, bbox.y+bbox.height},{bbox.x+bbox.width, bbox.y+bbox.height-L},col,2);
                cv::circle(arFrame, targetPoint, 4, col, -1, cv::LINE_AA);
                std::ostringstream ds;
                ds << std::fixed << std::setprecision(1) << "Z: " << zDyn << " cm";
                cv::putText(arFrame, "TARGET LOCKED", {bbox.x, bbox.y-20},
                            cv::FONT_HERSHEY_DUPLEX, 0.5, col, 1, cv::LINE_AA);
                cv::putText(arFrame, ds.str(), {bbox.x, bbox.y-5},
                            cv::FONT_HERSHEY_DUPLEX, 0.5, col, 1, cv::LINE_AA);
            } else {
                cv::Point sc  = {arFrame.cols/2, arFrame.rows/2};
                cv::Scalar gc = {150,150,150};
                cv::circle(arFrame, sc, 60, gc, 1, cv::LINE_AA);
                static double angle = 0.0; angle += 0.1;
                cv::line(arFrame, sc, {sc.x + int(60*std::cos(angle)), sc.y + int(60*std::sin(angle))}, gc, 1);
                cv::putText(arFrame, "SEARCHING...", {sc.x-50, sc.y-75},
                            cv::FONT_HERSHEY_DUPLEX, 0.5, gc, 1, cv::LINE_AA);
            }
            cv::imshow("2 - Efecto Realidad Aumentada", arFrame);
        }

        // ── Dashboard grid ────────────────────────────────────────────────────
        {
            cv::Mat stereoView;
            cv::hconcat(rL, rR, stereoView);
            for (int y = 0; y < stereoView.rows; y += 40)
                cv::line(stereoView, {0,y}, {stereoView.cols,y}, {0,255,0}, 1);

            cv::Mat t1 = buildDashboardTile(claheView,   320, 240); DebugViews::label(t1, "00 - CLAHE");
            cv::Mat t2 = buildDashboardTile(rawVis,      320, 240); DebugViews::label(t2, "01 - RAW SGBM");
            cv::Mat t3 = buildDashboardTile(wlsVis,      320, 240); DebugViews::label(t3, "02 - WLS FILTER");
            cv::Mat t4 = buildDashboardTile(stereoView,  640, 240); DebugViews::label(t4, "RECTIFIED PAIR (Epipolar Check)");
            cv::Mat t5 = buildDashboardTile(temporalVis, 320, 240); DebugViews::label(t5, "03 - TEMPORAL (FINAL)");

            cv::Mat topRow, bottomRow, dashboard;
            cv::hconcat(std::vector<cv::Mat>{t1, t2, t3}, topRow);
            cv::hconcat(t4, t5, bottomRow);
            cv::vconcat(topRow, bottomRow, dashboard);
            cv::imshow(winName, dashboard);
        }

        if (cv::waitKey(1) == 27) break;
    }
    cv::destroyWindow(winName);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    // ── Pipeline config ───────────────────────────────────────────────────────
    CalibrationConfig config;
    config.boardSize    = {9, 6};
    config.squareSizeM  = 0.025f;
    config.targetPairs  = 30;
    config.datasetDir   = "calib_pairs";
    config.leftYaml     = "calib_pairs/calib/left.yaml";
    config.rightYaml    = "calib_pairs/calib/right.yaml";
    config.stereoYaml   = "calib_pairs/calib/stereo.yaml";

    // ── Camera streams ────────────────────────────────────────────────────────
    bool camerasRunning = false;
    CameraStream cam1, cam2;
    std::thread t1, t2;

    auto ensureCams = [&]() {
        if (camerasRunning) return;
        std::cout << "\n  Starting camera streams...\n";
        t1 = std::thread(streamCamera, "http://192.168.18.111:81/stream", std::ref(cam1));
        t2 = std::thread(streamCamera, "http://192.168.18.112:81/stream", std::ref(cam2));
        waitForCameras(cam1, cam2);
        camerasRunning = true;
    };

    // ── Menu loop ─────────────────────────────────────────────────────────────
    while (true) {
        PipelineStatus st = PipelineStatus::check(config);
        printMenu(st);

        std::string input;
        std::getline(std::cin, input);
        input.erase(0, input.find_first_not_of(" \t"));
        if (input.empty()) continue;

        if (input == "0") { std::cout << "\n  Goodbye.\n\n"; break; }

        if (input == "1") {
            ensureCams();
            runCalibrationMode(cam1, cam2, config);
            continue;
        }
        if (input == "2") {
            if (!st.datasetReady()) { std::cout << "\n  Dataset not ready.\n"; continue; }
            runMonoCalibration(config);
            std::cout << "\nPress ENTER..."; std::getline(std::cin, input);
            continue;
        }
        if (input == "3") {
            if (!st.monoReady()) { std::cout << "\n  Run mono calibration first.\n"; continue; }
            runStereoCalibration(config);
            std::cout << "\nPress ENTER..."; std::getline(std::cin, input);
            continue;
        }
        if (input == "4") {
            if (!st.stereoReady()) { std::cout << "\n  Run stereo calibration first.\n"; continue; }
            ensureCams();
            runLiveRectifiedPreview(cam1, cam2, config);
            continue;
        }
        if (input == "5") {
            if (!st.stereoReady()) { std::cout << "\n  Run stereo calibration first.\n"; continue; }
            runEpipolarDatasetCheck(config);
            continue;
        }
        if (input == "6") {
            if (!st.stereoReady()) { std::cout << "\n  Run stereo calibration first.\n"; continue; }
            ensureCams();
            runLiveDisparity(cam1, cam2, config);
            continue;
        }

        std::cout << "\n  Unknown option.\n";
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    if (camerasRunning) {
        running = false;
        cv::destroyAllWindows();
        t1.join();
        t2.join();
    }
    return 0;
}