#pragma once
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

// ─────────────────────────────────────────────────────────────────────────────
// Log — logger de diagnóstico estratégico
//
// Reglas de diseño:
//   - Etiquetar cada mensaje con [NIVEL][TAG] para poder filtrar con grep
//   - Limitar la frecuencia de mensajes repetitivos (fallos de detección) para evitar spam
//   - Nunca registrar log en cada frame a menos que algo haya cambiado o superado un umbral
//   - Las marcas de tiempo son relativas al inicio del programa
//
// Uso:
//   Log::info("Detector", "Esquinas encontradas en cámara IZQUIERDA");
//   Log::warn("Session",  "Omitiendo captura duplicada (cooldown activo)");
//   Log::detection("IZQ", found, brightness, claheUsed);
// ─────────────────────────────────────────────────────────────────────────────
namespace Log {

// ── Funciones auxiliares internas (no son parte de la API pública) ───────────
namespace detail {

inline std::string timestamp() {
    static const auto start = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::steady_clock::now() - start;
    double secs  = std::chrono::duration<double>(elapsed).count();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << std::setw(7) << secs << "s";
    return ss.str();
}

// Retorna true si ha pasado suficiente tiempo desde la última llamada para esta llave.
// Usado para limitar advertencias repetitivas (ej. "muy oscuro" cada frame).
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

// ── API Pública ───────────────────────────────────────────────────────────────

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

// Llamada una vez por intento de detección — pero LIMITADA para que solo imprima
// cuando el resultado cambie o cada N segundos (evita spam por cada frame).
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
        ss << "✓ ENCONTRADO " << cornersFound << "/" << cornersExpected << " esquinas"
           << " | brillo=" << std::fixed << std::setprecision(1) << brightness
           << (claheUsed ? " | CLAHE=ON" : " | CLAHE=OFF");
        info(cam, ss.str());
    } else {
        ss << "✗ NO ENCONTRADO"
           << " | brillo=" << std::fixed << std::setprecision(1) << brightness;
        if (brightness < 60.0)  ss << " ← MUY OSCURO (necesita >60)";
        if (brightness > 220.0) ss << " ← MUY BRILLANTE / sobreexpuesto";
        if (claheUsed)          ss << " | CLAHE=ON (aun asi fallo)";
        else                    ss << " | CLAHE=OFF (intente activarlo)";
        warn(cam, ss.str());
    }
}

// Llamado una vez por cada par guardado.
inline void capture(int done, int total, const std::string& leftPath) {
    std::ostringstream ss;
    ss << "Par " << std::setw(2) << std::setfill('0') << done
       << "/" << total << " guardado → " << leftPath;
    info("Sesion", ss.str());
}

// Llamado después de que calibrateCamera o stereoCalibrate terminen.
inline void calibResult(const std::string& phase, double rpe,
                        int imagesUsed, int imagesTotal) {
    std::ostringstream ss;
    ss << "Error de reproyeccion RMS = " << std::fixed << std::setprecision(4)
       << rpe << " px"
       << " | imagenes usadas: " << imagesUsed << "/" << imagesTotal;

    if      (rpe < 0.3)  ss << "  ✓✓ Excelente";
    else if (rpe < 0.5)  ss << "  ✓  Bueno";
    else if (rpe < 1.0)  ss << "  △  Aceptable (considere recapturar)";
    else                 ss << "  ✗  Deficiente — se recomienda recapturar";

    info(phase, ss.str());
}

// Llamado al cargar un archivo YAML.
inline void yamlLoaded(const std::string& path, bool success) {
    if (success) info("YAML", "Cargado: " + path);
    else         error("YAML", "Fallo al cargar: " + path);
}

// Llamado al guardar un archivo YAML.
inline void yamlSaved(const std::string& path) {
    info("YAML", "Guardado: " + path);
}

} // namespace Log