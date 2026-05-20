#include <opencv2/opencv.hpp>
#include "CameraStream.hpp"
#include "StreamWorker.hpp"
#include "CalibrationConfig.hpp"
#include "CalibrationMode.hpp"
#include <thread>
#include <string>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// Preview loop  (unchanged from your original architecture)
// ─────────────────────────────────────────────────────────────────────────────
static void runPreviewMode(CameraStream& cam1, CameraStream& cam2) {
    std::cout << "[Preview] ESC = exit | C = switch to calibration mode\n";

    while (true) {
        cv::Mat f1, f2;
        {
            std::lock_guard<std::mutex> lock(cam1.frame_mtx);
            if (!cam1.frame.empty()) f1 = cam1.frame.clone();
        }
        {
            std::lock_guard<std::mutex> lock(cam2.frame_mtx);
            if (!cam2.frame.empty()) f2 = cam2.frame.clone();
        }

        auto overlay = [](cv::Mat& f, const CameraStream& cam, const std::string& name) {
            auto now   = std::chrono::steady_clock::now();
            double age = std::chrono::duration<double, std::milli>(
                             now - cam.timestamp).count();
            cv::putText(f, name,
                        cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(255,255,0), 2);
            cv::putText(f, "FPS: " + std::to_string(static_cast<int>(cam.fps)),
                        cv::Point(20, 58), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar(0,255,0), 2);
            cv::putText(f, "Age: " + std::to_string(static_cast<int>(age)) + "ms",
                        cv::Point(20, 84), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar(0,255,255), 2);
            cv::putText(f, "C=calibrate  ESC=exit",
                        cv::Point(10, f.rows - 10), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(200,200,200), 1);
        };

        if (!f1.empty()) { overlay(f1, cam1, "LEFT");  cv::imshow("Cam LEFT",  f1); }
        if (!f2.empty()) { overlay(f2, cam2, "RIGHT"); cv::imshow("Cam RIGHT", f2); }

        int key = cv::waitKey(1);
        if (key == 27)           break;   // ESC
        if (key == 'c' || key == 'C') {
            cv::destroyAllWindows();
            return;   // caller will switch to calibration
        }
    }

    running = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {

    // ── Mode selection via CLI argument ──────────────────────────────────────
    // ./StereoVision            → preview mode first
    // ./StereoVision --calibrate → jump straight to calibration
    bool startCalibration = (argc > 1 && std::string(argv[1]) == "--calibrate");

    // ── Start camera streams (unchanged from your original) ──────────────────
    CameraStream cam1, cam2;

    std::thread t1(streamCamera, "http://192.168.18.112:81/stream", std::ref(cam1));
    std::thread t2(streamCamera, "http://192.168.18.111:81/stream", std::ref(cam2));

    // Brief warm-up: wait until both cameras send at least one frame
    std::cout << "[Main] Waiting for cameras...\n";
    while (true) {
        bool c1ok, c2ok;
        { std::lock_guard<std::mutex> l(cam1.frame_mtx); c1ok = !cam1.frame.empty(); }
        { std::lock_guard<std::mutex> l(cam2.frame_mtx); c2ok = !cam2.frame.empty(); }
        if (c1ok && c2ok) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::cout << "[Main] Both cameras online.\n";

    // ── Calibration configuration ─────────────────────────────────────────────
    CalibrationConfig config;
    // config.boardSize   = {9, 6};   // already the default — change if needed
    // config.squareSizeM = 0.025f;   // 25 mm squares
    // config.targetPairs = 30;
    // config.outputDir   = "calib_pairs";

    // ── Run modes ─────────────────────────────────────────────────────────────
    if (!startCalibration) {
        runPreviewMode(cam1, cam2);   // returns when user presses C or ESC
    }

    if (running) {
        // User pressed C (or passed --calibrate): enter calibration
        runCalibrationMode(cam1, cam2, config);
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    running = false;
    cv::destroyAllWindows();
    t1.join();
    t2.join();

    return 0;
}
