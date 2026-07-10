#pragma once
#include "CameraStream.hpp"
#include "CalibrationConfig.hpp"

// Vista previa de cámaras rectificadas con líneas epipolares
void runLiveRectifiedPreview(CameraStream& cam1, CameraStream& cam2, const CalibrationConfig& config);

// Verificación de líneas epipolares en el dataset estático
void runEpipolarDatasetCheck(const CalibrationConfig& config);
