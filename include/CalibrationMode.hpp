#pragma once
#include "CameraStream.hpp"
#include "CalibrationConfig.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// runCalibrationMode
//
// Bucle interactivo autónomo para captura de imágenes de calibración estéreo.
// Lee frames de ambos streams, ejecuta detección de tablero en cada uno,
// muestra vista lado a lado con información y captura pares presionando ESPACIO.
//
// Retorna cuando:
//   - Se presiona ESC, O
//   - session.isComplete() (se alcanza el objetivo de pares)
//
// cam1 = Stream cámara IZQUIERDA (ya corriendo en su propio hilo)
// cam2 = Stream cámara DERECHA (ya corriendo en su propio hilo)
// ─────────────────────────────────────────────────────────────────────────────
void runCalibrationMode(CameraStream& cam1, CameraStream& cam2,
                        const CalibrationConfig& config);
