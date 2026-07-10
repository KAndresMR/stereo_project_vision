#pragma once
#include "CameraStream.hpp"
#include "CalibrationConfig.hpp"

// Ejecuta el modo de disparidad en vivo (SGBM) con seguimiento de objetivos AR y panel de control
void runLiveDisparity(CameraStream& cam1, CameraStream& cam2, const CalibrationConfig& config);
