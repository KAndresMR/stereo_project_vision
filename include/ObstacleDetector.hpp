#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
// ObstacleDetector
//
// Uses a depth map (CV_32F, meters) to detect objects in configurable
// distance zones and generate alerts.
//
// Design:
//   - Zones are user-defined bands of depth (e.g. DANGER: 0–0.5m, WARN: 0.5–1m)
//   - Detection triggers when enough pixels fall within a zone in the ROI
//   - Minimum pixel threshold prevents noise from triggering false alerts
//   - Optional morphological filtering on the zone mask reduces salt-and-pepper
//
// Usage:
//   ObstacleDetector od;
//   od.addZone({"DANGER", 0.0f, 0.5f, cv::Scalar(0,0,220), 0.05f});
//   od.addZone({"WARN",   0.5f, 1.2f, cv::Scalar(0,165,255), 0.03f});
//
//   auto alerts = od.detect(depthMap);
//   cv::Mat vis  = od.visualize(depthMap, frame);
// ─────────────────────────────────────────────────────────────────────────────
class ObstacleDetector {
public:

    struct Zone {
        std::string  name;
        float        minM;         // near boundary (meters)
        float        maxM;         // far boundary (meters)
        cv::Scalar   color;        // BGR for visualization
        float        minCoverage;  // fraction of ROI pixels to trigger [0,1]
    };

    struct Alert {
        std::string zoneName;
        float       coverage;     // fraction of ROI pixels in this zone
        float       minDepth;     // nearest detected pixel (m)
        bool        triggered;
    };

    struct Config {
        cv::Rect roi;             // region of interest (empty = full frame)
        bool     morphFilter = true;     // apply open+close to zone mask
        int      morphKernelSize = 5;    // morphological kernel size
        bool     drawZoneMask = true;    // overlay zone masks on visualization
        bool     drawDepthText = true;   // show depth values on visualization
    };

    explicit ObstacleDetector(const Config& cfg = {});

    void addZone(const Zone& z);
    void clearZones();

    // Process a depth map. Returns one Alert per zone.
    std::vector<Alert> detect(const cv::Mat& depthM) const;

    // Visualize depth zones overlaid on a reference color frame.
    cv::Mat visualize(const cv::Mat& depthM,
                      const cv::Mat& colorFrame,
                      const std::vector<Alert>& alerts) const;

    // Draw HUD alert panel on the visualization image.
    void drawAlertPanel(cv::Mat& vis,
                        const std::vector<Alert>& alerts) const;

private:
    Config            cfg_;
    std::vector<Zone> zones_;

    cv::Mat getROIMask(const cv::Mat& depthM) const;
    cv::Mat applyMorphology(const cv::Mat& mask) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// Implementation (header-only for simplicity)
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <iomanip>
#include <sstream>

inline ObstacleDetector::ObstacleDetector(const Config& cfg) : cfg_(cfg) {}

inline void ObstacleDetector::addZone(const Zone& z) { zones_.push_back(z); }
inline void ObstacleDetector::clearZones() { zones_.clear(); }

inline cv::Mat ObstacleDetector::getROIMask(const cv::Mat& depthM) const {
    cv::Mat mask = cv::Mat::ones(depthM.size(), CV_8U) * 255;
    if (!cfg_.roi.empty()) {
        mask = cv::Mat::zeros(depthM.size(), CV_8U);
        mask(cfg_.roi) = 255;
    }
    return mask;
}

inline cv::Mat ObstacleDetector::applyMorphology(const cv::Mat& mask) const {
    if (!cfg_.morphFilter) return mask;
    cv::Mat k = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(cfg_.morphKernelSize, cfg_.morphKernelSize));
    cv::Mat result;
    cv::morphologyEx(mask, result, cv::MORPH_OPEN,  k);  // remove noise
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, k); // fill gaps
    return result;
}

inline std::vector<ObstacleDetector::Alert>
ObstacleDetector::detect(const cv::Mat& depthM) const {
    std::vector<Alert> alerts;
    if (depthM.empty() || zones_.empty()) return alerts;

    cv::Mat roiMask = getROIMask(depthM);

    // Total valid pixels in ROI (depth > 0)
    cv::Mat validInROI = (depthM > 0.0f) & roiMask;
    int totalValid = cv::countNonZero(validInROI);
    if (totalValid == 0) return alerts;

    for (const auto& zone : zones_) {
        // Pixels within this depth zone AND inside ROI AND valid
        cv::Mat zoneMask = (depthM >= zone.minM) &
                           (depthM <  zone.maxM) &
                           roiMask & (depthM > 0.0f);

        zoneMask = applyMorphology(zoneMask);

        int zonePixels = cv::countNonZero(zoneMask);
        float coverage = static_cast<float>(zonePixels) / totalValid;

        // Find minimum depth within zone
        float minDepth = zone.maxM;
        if (zonePixels > 0) {
            double minVal;
            cv::minMaxLoc(depthM, &minVal, nullptr, nullptr, nullptr, zoneMask);
            minDepth = static_cast<float>(minVal);
        }

        alerts.push_back({
            zone.name,
            coverage,
            minDepth,
            coverage >= zone.minCoverage
        });
    }

    return alerts;
}

inline cv::Mat ObstacleDetector::visualize(
        const cv::Mat& depthM,
        const cv::Mat& colorFrame,
        const std::vector<Alert>& alerts) const {

    cv::Mat vis = colorFrame.clone();
    if (vis.empty()) {
        vis = cv::Mat::zeros(depthM.size(), CV_8UC3);
    }

    if (cfg_.drawZoneMask && !zones_.empty()) {
        cv::Mat overlay = vis.clone();

        for (size_t i = 0; i < zones_.size() && i < alerts.size(); ++i) {
            const auto& zone  = zones_[i];
            const auto& alert = alerts[i];

            cv::Mat zoneMask = (depthM >= zone.minM) &
                               (depthM <  zone.maxM) &
                               (depthM > 0.0f);

            // Color overlay on zone pixels
            overlay.setTo(zone.color, zoneMask);
        }

        // Blend overlay with original
        cv::addWeighted(vis, 0.6, overlay, 0.4, 0, vis);
    }

    // ROI rectangle
    if (!cfg_.roi.empty()) {
        cv::rectangle(vis, cfg_.roi, cv::Scalar(200, 200, 200), 1);
    }

    drawAlertPanel(vis, alerts);

    return vis;
}

inline void ObstacleDetector::drawAlertPanel(
        cv::Mat& vis,
        const std::vector<Alert>& alerts) const {

    int panelX = 10;
    int panelY = vis.rows - 30 - (int)alerts.size() * 28;
    int panelW = 280;
    int panelH = (int)alerts.size() * 28 + 10;

    // Dark background panel
    cv::rectangle(vis,
        cv::Point(panelX - 4, panelY - 4),
        cv::Point(panelX + panelW, panelY + panelH),
        cv::Scalar(20, 20, 20), cv::FILLED);

    for (size_t i = 0; i < alerts.size() && i < zones_.size(); ++i) {
        const auto& a = alerts[i];
        const auto& z = zones_[i];

        int y = panelY + (int)i * 28 + 18;

        // Alert indicator
        cv::Scalar indicatorColor = a.triggered
            ? z.color
            : cv::Scalar(80, 80, 80);

        cv::circle(vis, {panelX + 8, y - 4}, 7, indicatorColor, cv::FILLED);

        // Zone label + metrics
        std::ostringstream ss;
        ss << z.name
           << "  " << std::fixed << std::setprecision(0) << (a.coverage * 100.0f) << "%"
           << "  min=" << std::setprecision(2) << a.minDepth << "m";

        cv::Scalar textColor = a.triggered
            ? cv::Scalar(255, 255, 255)
            : cv::Scalar(140, 140, 140);

        cv::putText(vis, ss.str(), {panelX + 20, y},
            cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1);
    }
}
