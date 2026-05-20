#pragma once
#include "CameraStream.hpp"
#include "CalibrationConfig.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// runCalibrationMode
//
// Self-contained interactive loop for stereo calibration image capture.
// Reads frames from both streams, runs chessboard detection on each,
// shows a side-by-side view with overlays, and captures pairs on SPACE.
//
// Returns when:
//   - ESC is pressed, OR
//   - session.isComplete() (targetPairs reached)
//
// cam1 = LEFT camera stream (already running in its own thread)
// cam2 = RIGHT camera stream (already running in its own thread)
// ─────────────────────────────────────────────────────────────────────────────
void runCalibrationMode(CameraStream& cam1, CameraStream& cam2,
                        const CalibrationConfig& config);
