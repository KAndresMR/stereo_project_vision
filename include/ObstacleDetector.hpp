#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
// ObstacleDetector
//
// Usa un mapa de profundidad (CV_32F, metros) para detectar objetos en zonas de
// distancia configurables y generar alertas.
//
// Diseño:
//   - Las zonas son bandas de profundidad definidas por el usuario (ej. PELIGRO: 0–0.5m)
//   - La detección se activa cuando suficientes píxeles caen dentro de una zona en la región de interés (ROI)
//   - Un umbral mínimo de píxeles evita que el ruido provoque falsas alarmas
//   - Filtrado morfológico opcional en la máscara de zona para reducir el ruido
//
// Uso:
//   ObstacleDetector od;
//   od.addZone({"PELIGRO", 0.0f, 0.5f, cv::Scalar(0,0,220), 0.05f});
//   od.addZone({"AVISO",   0.5f, 1.2f, cv::Scalar(0,165,255), 0.03f});
//
//   auto alerts = od.detect(depthMap);
//   cv::Mat vis  = od.visualize(depthMap, frame);
// ─────────────────────────────────────────────────────────────────────────────
class ObstacleDetector {
public:

    struct Zone {
        std::string  name;
        float        minM;         // límite cercano (metros)
        float        maxM;         // límite lejano (metros)
        cv::Scalar   color;        // color BGR para visualización
        float        minCoverage;  // fracción de píxeles ROI para activarse [0,1]
    };

    struct Alert {
        std::string zoneName;
        float       coverage;     // fracción de píxeles ROI en esta zona
        float       minDepth;     // píxel más cercano detectado (m)
        bool        triggered;
    };

    struct Config {
        cv::Rect roi;             // región de interés (vacío = todo el frame)
        bool     morphFilter = true;     // aplicar open+close a la máscara de zona
        int      morphKernelSize = 5;    // tamaño del kernel morfológico
        bool     drawZoneMask = true;    // superponer máscaras de zona en la visualización
        bool     drawDepthText = true;   // mostrar valores de profundidad en la visualización
    };

    explicit ObstacleDetector(const Config& cfg = {});

    void addZone(const Zone& z);
    void clearZones();

    // Procesa un mapa de profundidad. Retorna una Alerta por zona.
    std::vector<Alert> detect(const cv::Mat& depthM) const;

    // Visualizar zonas de profundidad superpuestas en un frame a color de referencia.
    cv::Mat visualize(const cv::Mat& depthM,
                      const cv::Mat& colorFrame,
                      const std::vector<Alert>& alerts) const;

    // Dibuja el panel de alertas HUD en la imagen de visualización.
    void drawAlertPanel(cv::Mat& vis,
                        const std::vector<Alert>& alerts) const;

private:
    Config            cfg_;
    std::vector<Zone> zones_;

    cv::Mat getROIMask(const cv::Mat& depthM) const;
    cv::Mat applyMorphology(const cv::Mat& mask) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// Implementación (solo en header por simplicidad)
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
    cv::morphologyEx(mask, result, cv::MORPH_OPEN,  k);  // remover ruido
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, k); // llenar huecos
    return result;
}

inline std::vector<ObstacleDetector::Alert>
ObstacleDetector::detect(const cv::Mat& depthM) const {
    std::vector<Alert> alerts;
    if (depthM.empty() || zones_.empty()) return alerts;

    cv::Mat roiMask = getROIMask(depthM);

    // Total de píxeles válidos en la ROI (profundidad > 0)
    cv::Mat validInROI = (depthM > 0.0f) & roiMask;
    int totalValid = cv::countNonZero(validInROI);
    if (totalValid == 0) return alerts;

    for (const auto& zone : zones_) {
        // Píxeles dentro de esta zona de profundidad Y dentro de ROI Y válidos
        cv::Mat zoneMask = (depthM >= zone.minM) &
                           (depthM <  zone.maxM) &
                           roiMask & (depthM > 0.0f);

        zoneMask = applyMorphology(zoneMask);

        int zonePixels = cv::countNonZero(zoneMask);
        float coverage = static_cast<float>(zonePixels) / totalValid;

        // Encontrar profundidad mínima dentro de la zona
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

            // Superposición de color en los píxeles de la zona
            overlay.setTo(zone.color, zoneMask);
        }

        // Mezclar superposición con original
        cv::addWeighted(vis, 0.6, overlay, 0.4, 0, vis);
    }

    // Rectángulo de ROI
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

    // Panel de fondo oscuro
    cv::rectangle(vis,
        cv::Point(panelX - 4, panelY - 4),
        cv::Point(panelX + panelW, panelY + panelH),
        cv::Scalar(20, 20, 20), cv::FILLED);

    for (size_t i = 0; i < alerts.size() && i < zones_.size(); ++i) {
        const auto& a = alerts[i];
        const auto& z = zones_[i];

        int y = panelY + (int)i * 28 + 18;

        // Indicador de alerta
        cv::Scalar indicatorColor = a.triggered
            ? z.color
            : cv::Scalar(80, 80, 80);

        cv::circle(vis, {panelX + 8, y - 4}, 7, indicatorColor, cv::FILLED);

        // Etiqueta de zona + métricas
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
