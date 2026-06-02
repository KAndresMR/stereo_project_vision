#pragma once
#include "CalibrationConfig.hpp"

// Runs the full calibration pipeline: Mono Calibration for Left and Right,
// followed immediately by Stereo Calibration.
void runFullCalibration(const CalibrationConfig& config);
