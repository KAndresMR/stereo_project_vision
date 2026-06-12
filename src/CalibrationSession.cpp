#include "CalibrationSession.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────

CalibrationSession::CalibrationSession(
    const CalibrationConfig& config)
    : config_(config)
{
    ensureDirectories();

    scanExistingDataset();

    std::cout
        << "[Session] Salida: "
        << fs::absolute(config_.datasetDir)
        << "\n";

    std::cout
        << "[Session] Pares existentes: "
        << pairCount_
        << "\n";

    std::cout
        << "[Session] Objetivo: "
        << config_.targetPairs
        << " pares\n";
}

void CalibrationSession::scanExistingDataset()
{
    namespace fs = std::filesystem;

    pairCount_ = 0;

    const std::string leftDir =
        config_.datasetDir + "/left";

    if (!fs::exists(leftDir))
        return;

    for (const auto& entry :
         fs::directory_iterator(leftDir))
    {
        if (!entry.is_regular_file())
            continue;

        if (entry.path().extension() != ".jpg")
            continue;

        ++pairCount_;
    }
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

    // Calidad JPEG 95 — suficientemente alta para preservar detalle de esquinas para calibración.
    // Evitar calidad 100 (duplica tamaño de archivo con beneficio mínimo).
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 95};

    bool ok = cv::imwrite(leftPath(idx),  left,  params)
           && cv::imwrite(rightPath(idx), right, params);

    if (ok) {
        pairCount_++;
        std::cout << "[Session] Par " << std::setw(2) << pairCount_
                  << "/" << config_.targetPairs
                  << " → " << leftPath(idx) << "\n";
    } else {
        std::cerr << "[Session] ERROR: no se pudo escribir el par " << idx << "\n";
    }

    return ok;
}
