#pragma once
#include "CameraStream.hpp"
#include "CalibrationConfig.hpp"

// Runs the live disparity (SGBM) mode with AR target tracking and dashboard
void runLiveDisparity(CameraStream& cam1, CameraStream& cam2, const CalibrationConfig& config);
