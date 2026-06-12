#include "CalibrationPipeline.hpp"
#include "MonoCalibrator.hpp"
#include "StereoCalibrator.hpp"
#include "DiagnosticLogger.hpp"
#include <iostream>

static void runMonoCalibration(const CalibrationConfig& config) {
    Log::separator("FASE 1 — Calibracion Monocular");
    MonoCalibrator mc(config);
    auto left  = mc.calibrate(MonoCalibrator::Side::LEFT);
    if (left.success)  mc.saveYAML(left,  MonoCalibrator::Side::LEFT);
    auto right = mc.calibrate(MonoCalibrator::Side::RIGHT);
    if (right.success) mc.saveYAML(right, MonoCalibrator::Side::RIGHT);
}

static void runStereoCalibration(const CalibrationConfig& config) {
    Log::separator("FASE 2 — Calibracion Estereo");
    StereoCalibrator stereo(config);
    auto result = stereo.calibrate();
    if (!result.success) { 
        Log::error("CalibrationPipeline", "La calibracion estereo fallo"); 
        return; 
    }
    stereo.saveYAML(result);
    Log::info("CalibrationPipeline", "YAML Estereo guardado");
}

void runFullCalibration(const CalibrationConfig& config) {
    runMonoCalibration(config);
    runStereoCalibration(config);
    std::cout << "\n  ¡Proceso de calibracion finalizado exitosamente!\n";
}
