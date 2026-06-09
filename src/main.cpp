#include <opencv2/opencv.hpp>
#include "CameraStream.hpp"
#include "CalibrationConfig.hpp"
#include "StreamWorker.hpp"
#include "CalibrationMode.hpp"
#include "CalibrationPipeline.hpp"
#include "DisparityMode.hpp"
#include "DiagnosticModes.hpp" // Included for diagnostics (hidden features)
#include "DiagnosticLogger.hpp"
#include <thread>
#include <iostream>
#include <string>
#include <filesystem>

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
        << "  │  1. Capture dataset                         │\n"
        << "  │  2. Run calibration (Mono + Stereo)         │\n"
        << "  │  3. Live disparity (SGBM)                   │\n"
        << "  │                                             │\n"
        << "  │  0. Exit                                    │\n"
        << "  └─────────────────────────────────────────────┘\n"
        << "\n  Choice: ";
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
        t1 = std::thread(streamCamera, "http://192.168.1.2:81/stream", std::ref(cam1)); 
        t2 = std::thread(streamCamera, "http://192.168.1.3:81/stream", std::ref(cam2));
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
            if (!st.datasetReady()) { std::cout << "\n  Dataset not ready (min 10 pairs required).\n"; continue; }
            runFullCalibration(config);
            std::cout << "\nPress ENTER..."; std::getline(std::cin, input);
            continue;
        }
        if (input == "3") {
            if (!st.stereoReady()) { std::cout << "\n  Run calibration first.\n"; continue; }
            ensureCams();
            runLiveDisparity(cam1, cam2, config);
            continue;
        }

        // ── HIDDEN DIAGNOSTIC OPTIONS (Uncomment / use directly if needed) ─────
        /*
        if (input == "8") {
            if (!st.stereoReady()) { std::cout << "\n  Run calibration first.\n"; continue; }
            runEpipolarDatasetCheck(config);
            continue;
        }
        if (input == "9") {
            if (!st.stereoReady()) { std::cout << "\n  Run calibration first.\n"; continue; }
            ensureCams();
            runLiveRectifiedPreview(cam1, cam2, config);
            continue;
        }
        */

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