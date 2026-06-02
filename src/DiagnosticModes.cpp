#include "DiagnosticModes.hpp"
#include "Rectifier.hpp"
#include "DiagnosticLogger.hpp"
#include <opencv2/opencv.hpp>

void runLiveRectifiedPreview(CameraStream& cam1, CameraStream& cam2, const CalibrationConfig& config) {
    Rectifier rect(config);
    if (!rect.compute()) { 
        Log::error("DiagnosticModes", "Rectification init failed"); 
        return; 
    }

    while (true) {
        cv::Mat f1 = grabFrame(cam1);
        cv::Mat f2 = grabFrame(cam2);
        if (!f1.empty() && !f2.empty()) {
            auto [rL, rR] = rect.rectify(f1, f2);
            cv::imshow("Live Rectified", rect.drawEpipolarLines(rL, rR));
        }
        if (cv::waitKey(1) == 27) break;
    }
    cv::destroyAllWindows();
}

void runEpipolarDatasetCheck(const CalibrationConfig& config) {
    Rectifier rect(config);
    if (!rect.compute()) { 
        Log::error("DiagnosticModes", "Rectification failed"); 
        return; 
    }
    rect.previewDataset();
}
