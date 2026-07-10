#pragma once
#include "CalibrationConfig.hpp"

// Ejecuta el pipeline completo de calibración: Calibración Monocular para Izquierda y Derecha,
// seguido inmediatamente de la Calibración Estéreo.
void runFullCalibration(const CalibrationConfig& config);
