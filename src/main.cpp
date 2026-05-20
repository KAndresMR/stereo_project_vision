#include <opencv2/opencv.hpp>
#include "CameraStream.hpp"
#include "StreamWorker.hpp"
#include "CalibrationConfig.hpp"
#include "CalibrationMode.hpp"
#include "MonoCalibrator.hpp"
#include "DiagnosticLogger.hpp"
#include <thread>
#include <iostream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// printUsage
// ─────────────────────────────────────────────────────────────────────────────
static void printUsage(const char* exe) {
    std::cout <<
        "\nUsage:\n"
        "  " << exe << " [mode]\n\n"
        "Modes:\n"
        "  (none)          Interactive menu\n"
        "  --preview       Stream both cameras, no calibration\n"
        "  --capture       Capture stereo chessboard pairs interactively\n"
        "  --mono-calib    Run individual calibration on captured dataset\n\n"
        "Pipeline order:\n"
        "  1. --capture        → calib_pairs/left/*.jpg + right/*.jpg\n"
        "  2. --mono-calib     → calib_pairs/calib/left.yaml + right.yaml\n"
        "  3. (next) --stereo-calib  [not yet implemented]\n"
        "  4. (next) --rectify       [not yet implemented]\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Preview loop (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────
static void runPreviewMode(CameraStream& cam1, CameraStream& cam2) {
    Log::info("Preview", "SPACE=capture mode  |  ESC=exit");

    while (true) {
        cv::Mat f1, f2;
        { std::lock_guard<std::mutex> l(cam1.frame_mtx);
          if (!cam1.frame.empty()) f1 = cam1.frame.clone(); }
        { std::lock_guard<std::mutex> l(cam2.frame_mtx);
          if (!cam2.frame.empty()) f2 = cam2.frame.clone(); }

        auto overlay = [](cv::Mat& f, const CameraStream& cam,
                          const std::string& label) {
            cv::putText(f, label, {10,28},
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0,255,255), 2);
            cv::putText(f, "FPS:" + std::to_string((int)cam.fps), {10,54},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0,255,0), 1);
            cv::putText(f, "ESC=exit", {10, f.rows-8},
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(180,180,180), 1);
        };

        if (!f1.empty()) { 
            std::cout << "Cam1: "
                << f1.cols << "x" << f1.rows
                << std::endl;
            overlay(f1, cam1, "LEFT");  cv::imshow("LEFT",  f1); 
        }
        if (!f2.empty()) { 
            std::cout << "Cam2: "
                << f2.cols << "x" << f2.rows
                << std::endl;
            overlay(f2, cam2, "RIGHT"); cv::imshow("RIGHT", f2); 
        
        
        }

        if (cv::waitKey(1) == 27) break;
    }
    cv::destroyAllWindows();
}

// ─────────────────────────────────────────────────────────────────────────────
// runMonoCalibration — Phase 1
//
// Calibrates LEFT and RIGHT cameras individually using the saved dataset.
// Cameras do NOT need to be running for this mode — it works purely from disk.
// ─────────────────────────────────────────────────────────────────────────────
static void runMonoCalibration(const CalibrationConfig& config) {

    Log::separator("PHASE 1 — Individual Camera Calibration");
    Log::info("Main", "Cameras NOT required for this mode (reads from disk)");

    MonoCalibrator mc(config);

    // ── Calibrate LEFT ────────────────────────────────────────────────────────
    auto leftResult = mc.calibrate(MonoCalibrator::Side::LEFT);
    if (leftResult.success) {
        mc.saveYAML(leftResult, MonoCalibrator::Side::LEFT);
    } else {
        Log::error("Main", "LEFT calibration failed. Check dataset.");
    }

    // ── Calibrate RIGHT ───────────────────────────────────────────────────────
    auto rightResult = mc.calibrate(MonoCalibrator::Side::RIGHT);
    if (rightResult.success) {
        mc.saveYAML(rightResult, MonoCalibrator::Side::RIGHT);
    } else {
        Log::error("Main", "RIGHT calibration failed. Check dataset.");
    }

    // ── Final status ──────────────────────────────────────────────────────────
    Log::separator("Summary");
    if (leftResult.success && rightResult.success) {
        Log::info("Main", "Both cameras calibrated successfully.");
        Log::info("Main", "LEFT  YAML → " + config.leftYaml);
        Log::info("Main", "RIGHT YAML → " + config.rightYaml);
        Log::info("Main", "Next step: run --stereo-calib");
    } else {
        Log::warn("Main", "One or both calibrations failed.");
        Log::warn("Main", "Tips:");
        Log::warn("Main", "  - Check that calib_pairs/left/ and right/ are not empty");
        Log::warn("Main", "  - Make sure boardSize in config matches your printed board");
        Log::warn("Main", "  - Re-run --capture if dataset looks bad");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// waitForCameras — blocks until both cameras have sent at least one frame
// ─────────────────────────────────────────────────────────────────────────────
static void waitForCameras(CameraStream& cam1, CameraStream& cam2) {
    Log::info("Main", "Waiting for both cameras...");
    while (true) {
        bool ok1, ok2;
        { std::lock_guard<std::mutex> l(cam1.frame_mtx); ok1 = !cam1.frame.empty(); }
        { std::lock_guard<std::mutex> l(cam2.frame_mtx); ok2 = !cam2.frame.empty(); }
        if (ok1 && ok2) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    Log::info("Main", "Both cameras online ✓");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {

    // ── Parse mode ────────────────────────────────────────────────────────────
    std::string mode = (argc > 1) ? std::string(argv[1]) : "";

    // ── Config (edit values here or load from file in a future iteration) ─────
    CalibrationConfig config;
    config.boardSize    = {9, 6};     // 9×6 physical squares → 8×5 inner corners
    config.squareSizeM  = 0.025f;     // 25 mm squares
    config.targetPairs  = 30;
    config.datasetDir   = "calib_pairs";
    config.leftYaml     = "calib_pairs/calib/left.yaml";
    config.rightYaml    = "calib_pairs/calib/right.yaml";
    config.stereoYaml   = "calib_pairs/calib/stereo.yaml";

    // ── Modes that do NOT need live cameras ───────────────────────────────────
    if (mode == "--mono-calib") {
        runMonoCalibration(config);
        return 0;
    }

    if (mode == "--help" || mode == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    // ── Modes that DO need live cameras ───────────────────────────────────────
    Log::info("Main", "Starting camera streams...");

    CameraStream cam1, cam2;
    std::thread t1(streamCamera, "http://192.168.18.112:81/stream", std::ref(cam1));
    std::thread t2(streamCamera, "http://192.168.18.111:81/stream", std::ref(cam2));

    waitForCameras(cam1, cam2);

    if (mode == "--preview") {
        runPreviewMode(cam1, cam2);
    } else if (mode == "--capture") {
        runCalibrationMode(cam1, cam2, config);
    } else {
        // Interactive menu
        printUsage(argv[0]);
        Log::info("Main", "No mode specified — entering preview. Press ESC to exit.");
        runPreviewMode(cam1, cam2);
    }

    running = false;
    cv::destroyAllWindows();
    t1.join();
    t2.join();

    return 0;
}
