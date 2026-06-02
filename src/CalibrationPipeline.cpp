#include "CalibrationPipeline.hpp"
#include "MonoCalibrator.hpp"
#include "StereoCalibrator.hpp"
#include "DiagnosticLogger.hpp"
#include <iostream>

static void runMonoCalibration(const CalibrationConfig& config) {
    Log::separator("PHASE 1 — Mono Calibration");
    MonoCalibrator mc(config);
    auto left  = mc.calibrate(MonoCalibrator::Side::LEFT);
    if (left.success)  mc.saveYAML(left,  MonoCalibrator::Side::LEFT);
    auto right = mc.calibrate(MonoCalibrator::Side::RIGHT);
    if (right.success) mc.saveYAML(right, MonoCalibrator::Side::RIGHT);
}

static void runStereoCalibration(const CalibrationConfig& config) {
    Log::separator("PHASE 2 — Stereo Calibration");
    StereoCalibrator stereo(config);
    auto result = stereo.calibrate();
    if (!result.success) { 
        Log::error("CalibrationPipeline", "Stereo calibration failed"); 
        return; 
    }
    stereo.saveYAML(result);
    Log::info("CalibrationPipeline", "Stereo YAML saved");
}

void runFullCalibration(const CalibrationConfig& config) {
    runMonoCalibration(config);
    runStereoCalibration(config);
    std::cout << "\n  Calibration process finished successfully!\n";
}
