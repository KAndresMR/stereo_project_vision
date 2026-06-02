#pragma once
#include "CameraStream.hpp"
#include "CalibrationConfig.hpp"

// Preview rectified cameras with epipolar lines
void runLiveRectifiedPreview(CameraStream& cam1, CameraStream& cam2, const CalibrationConfig& config);

// Epipolar line check on the static dataset
void runEpipolarDatasetCheck(const CalibrationConfig& config);
