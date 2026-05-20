#pragma once
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

// ─────────────────────────────────────────────────────────────────────────────
// Log — strategic diagnostic logger
//
// Design rules:
//   - Tag every message with [LEVEL][TAG] so you can grep specific subsystems
//   - Throttle repetitive messages (detection failures) to avoid console spam
//   - Never log per-frame unless something changed or a threshold was crossed
//   - Timestamps are relative to program start (easier to read than wall clock)
//
// Usage:
//   Log::info("Detector", "Corners found on LEFT camera");
//   Log::warn("Session",  "Skipping duplicate capture (cooldown active)");
//   Log::detection("LEFT", found, brightness, claheUsed);
// ─────────────────────────────────────────────────────────────────────────────
namespace Log {

// ── Internal helpers (not part of the public API) ───────────────────────────
namespace detail {

inline std::string timestamp() {
    static const auto start = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::steady_clock::now() - start;
    double secs  = std::chrono::duration<double>(elapsed).count();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << std::setw(7) << secs << "s";
    return ss.str();
}

// Returns true if enough time has passed since last call for this key.
// Used to throttle repetitive warnings (e.g. "too dark" every frame).
inline bool throttle(const std::string& key, double intervalSeconds = 2.0) {
    using Clock = std::chrono::steady_clock;
    static std::unordered_map<std::string,
           std::chrono::steady_clock::time_point> lastPrint;
    auto now = Clock::now();
    auto it  = lastPrint.find(key);
    if (it == lastPrint.end() ||
        std::chrono::duration<double>(now - it->second).count() >= intervalSeconds) {
        lastPrint[key] = now;
        return true;
    }
    return false;
}

} // namespace detail

// ── Public API ───────────────────────────────────────────────────────────────

inline void info(const std::string& tag, const std::string& msg) {
    std::cout << "[" << detail::timestamp() << "][INFO ][" << tag << "] " << msg << "\n";
}

inline void warn(const std::string& tag, const std::string& msg) {
    std::cout << "[" << detail::timestamp() << "][WARN ][" << tag << "] " << msg << "\n";
}

inline void error(const std::string& tag, const std::string& msg) {
    std::cerr << "[" << detail::timestamp() << "][ERROR][" << tag << "] " << msg << "\n";
}

inline void separator(const std::string& label = "") {
    if (label.empty()) {
        std::cout << "  ─────────────────────────────────────────────────\n";
    } else {
        std::cout << "\n  ══════ " << label << " ══════\n\n";
    }
}

// Called once per detection attempt — but THROTTLED so it only prints
// when the result changes or every N seconds (avoids per-frame spam).
inline void detection(const std::string& cam,
                      bool   found,
                      double brightness,
                      bool   claheUsed,
                      int    cornersFound = 0,
                      int    cornersExpected = 0) {

    std::string key = cam + (found ? "_ok" : "_fail");
    if (!detail::throttle(key, found ? 3.0 : 2.0)) return;

    std::ostringstream ss;
    if (found) {
        ss << "✓ FOUND " << cornersFound << "/" << cornersExpected << " corners"
           << " | brightness=" << std::fixed << std::setprecision(1) << brightness
           << (claheUsed ? " | CLAHE=ON" : " | CLAHE=OFF");
        info(cam, ss.str());
    } else {
        ss << "✗ NOT FOUND"
           << " | brightness=" << std::fixed << std::setprecision(1) << brightness;
        if (brightness < 60.0)  ss << " ← TOO DARK (need >60)";
        if (brightness > 220.0) ss << " ← TOO BRIGHT / overexposed";
        if (claheUsed)          ss << " | CLAHE=ON (still failed)";
        else                    ss << " | CLAHE=OFF (try enabling it)";
        warn(cam, ss.str());
    }
}

// Called once per saved pair.
inline void capture(int done, int total, const std::string& leftPath) {
    std::ostringstream ss;
    ss << "Pair " << std::setw(2) << std::setfill('0') << done
       << "/" << total << " saved → " << leftPath;
    info("Session", ss.str());
}

// Called after calibrateCamera or stereoCalibrate finishes.
inline void calibResult(const std::string& phase, double rpe,
                        int imagesUsed, int imagesTotal) {
    std::ostringstream ss;
    ss << "RMS reprojection error = " << std::fixed << std::setprecision(4)
       << rpe << " px"
       << " | images used: " << imagesUsed << "/" << imagesTotal;

    if      (rpe < 0.3)  ss << "  ✓✓ Excellent";
    else if (rpe < 0.5)  ss << "  ✓  Good";
    else if (rpe < 1.0)  ss << "  △  Acceptable (consider recapturing)";
    else                 ss << "  ✗  Poor — recapture recommended";

    info(phase, ss.str());
}

// Called when loading a YAML file.
inline void yamlLoaded(const std::string& path, bool success) {
    if (success) info("YAML", "Loaded: " + path);
    else         error("YAML", "Failed to load: " + path);
}

// Called when saving a YAML file.
inline void yamlSaved(const std::string& path) {
    info("YAML", "Saved: " + path);
}

} // namespace Log