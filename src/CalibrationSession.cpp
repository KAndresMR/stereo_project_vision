#include "CalibrationSession.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────

CalibrationSession::CalibrationSession(const CalibrationConfig& config)
    : config_(config) {
    ensureDirectories();
    std::cout << "[Session] Output: " << fs::absolute(config_.datasetDir) << "\n";
    std::cout << "[Session] Target: " << config_.targetPairs << " pairs\n";
}

// ─────────────────────────────────────────────────────────────────────────────

void CalibrationSession::ensureDirectories() const {
    fs::create_directories(config_.datasetDir + "/left");
    fs::create_directories(config_.datasetDir + "/right");
}

// ─────────────────────────────────────────────────────────────────────────────

std::string CalibrationSession::leftPath(int n) const {
    std::ostringstream ss;
    ss << config_.datasetDir << "/left/left_"
       << std::setw(2) << std::setfill('0') << n << ".jpg";
    return ss.str();
}

std::string CalibrationSession::rightPath(int n) const {
    std::ostringstream ss;
    ss << config_.datasetDir << "/right/right_"
       << std::setw(2) << std::setfill('0') << n << ".jpg";
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────

bool CalibrationSession::savePair(const cv::Mat& left, const cv::Mat& right) {
    int idx = pairCount_ + 1;

    // JPEG quality 95 — high enough to preserve corner detail for calibration.
    // Avoid quality 100 (doubles file size with minimal benefit).
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 95};

    bool ok = cv::imwrite(leftPath(idx),  left,  params)
           && cv::imwrite(rightPath(idx), right, params);

    if (ok) {
        pairCount_++;
        std::cout << "[Session] Pair " << std::setw(2) << pairCount_
                  << "/" << config_.targetPairs
                  << " → " << leftPath(idx) << "\n";
    } else {
        std::cerr << "[Session] ERROR: could not write pair " << idx << "\n";
    }

    return ok;
}
