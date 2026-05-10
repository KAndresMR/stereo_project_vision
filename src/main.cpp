#include <opencv2/opencv.hpp>
#include <thread>

#include "CameraStream.hpp"
#include "StreamWorker.hpp"


// ─── BUG FIX 1: definir los marcadores JPEG ───────────────────────────────────
// SOI = Start of Image (FF D8), EOI = End of Image (FF D9)
const uchar JPEG_SOI[] = {0xFF, 0xD8};
const uchar JPEG_EOI[] = {0xFF, 0xD9};

std::atomic<bool> running(true);

struct CameraStream {
    std::vector<uchar> buffer;
    cv::Mat frame;
    std::mutex frame_mtx;   // protege frame
    std::mutex buffer_mtx;  // protege buffer
};

// ─── BUG FIX 2: retornar 0 cuando running=false aborta curl_easy_perform ──────
// Si el callback retorna algo distinto al tamaño esperado, libcurl cancela la
// transferencia limpiamente → el hilo que llama perform() puede terminar.
size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    if (!running) return 0; // Esto aborta la transferencia

    auto* stream = static_cast<CameraStream*>(userdata);
    size_t total = size * nmemb;
    {
        std::lock_guard<std::mutex> lock(stream->buffer_mtx);
        stream->buffer.insert(stream->buffer.end(), ptr, ptr + total);
    }
    return total;
}

void streamCamera(const std::string& url, CameraStream& stream) {
    CURL* curl = curl_easy_init();
    if (!curl) return;

    curl_easy_setopt(curl, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &stream);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL,      1L);
    // Sin timeout: el stream MJPEG es infinito mientras la cámara funcione.
    // curl_easy_perform bloqueará hasta que el callback retorne 0 (cuando
    // running=false) o hasta que la cámara cierre la conexión.

    // ─── BUG FIX 3: curl corre en ESTE hilo, el procesamiento en uno aparte ───
    // En tu código original, curl corría en curlThread y se llamaba
    // curl_easy_cleanup() antes de que ese hilo terminara → UB/crash.
    // Aquí: curl_easy_perform bloquea este hilo. El procesamiento de frames
    // va en un hilo separado. Cuando running=false, el callback aborta perform,
    // este hilo desbloquea, y después join() del hilo de proceso.

    std::thread processThread([&]() {
        while (running) {
            std::vector<uchar> localBuffer;
            {
                std::lock_guard<std::mutex> lock(stream.buffer_mtx);
                localBuffer = stream.buffer; // copia segura
            }

            // Buscar SOI
            auto soi = std::search(localBuffer.begin(), localBuffer.end(),
                                    JPEG_SOI, JPEG_SOI + 2);
            if (soi == localBuffer.end()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            // ─── BUG FIX 4: buscar EOI *después* del SOI, no desde el inicio ─
            // Si hay basura con FF D9 antes del SOI, tu código original rompía.
            auto eoi = std::search(soi + 2, localBuffer.end(),
                                    JPEG_EOI, JPEG_EOI + 2);
            if (eoi == localBuffer.end()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            // Extraer y decodificar el JPEG
            std::vector<uchar> jpg(soi, eoi + 2);
            cv::Mat img = cv::imdecode(jpg, cv::IMREAD_COLOR);
            if (!img.empty()) {
                std::lock_guard<std::mutex> lock(stream.frame_mtx);
                stream.frame = img;
            }

            // Descartar del buffer lo que ya procesamos
            size_t consumed = (eoi - localBuffer.begin()) + 2;
            {
                std::lock_guard<std::mutex> lock(stream.buffer_mtx);
                // Verificar que el buffer real no sea más corto que consumed
                // (poco probable pero defensivo)
                if (consumed <= stream.buffer.size()) {
                    stream.buffer.erase(stream.buffer.begin(),
                                        stream.buffer.begin() + consumed);
                }
            }
        }
    });

    curl_easy_perform(curl); // Bloquea hasta abort (running=false) o error
    curl_easy_cleanup(curl); // Seguro: perform ya terminó

    processThread.join();
}

int main() {
    CameraStream cam1, cam2;

    std::thread t1(streamCamera, "http://192.168.18.111:81/stream", std::ref(cam1));
    std::thread t2(streamCamera, "http://192.168.18.112:81/stream", std::ref(cam2));

    while (true) {

        cv::Mat f1;
        cv::Mat f2;

        {
            std::lock_guard<std::mutex> lock(
                cam1.frame_mtx
            );

            if (!cam1.frame.empty()) {
                f1 = cam1.frame.clone();
            }
        }

        {
            std::lock_guard<std::mutex> lock(
                cam2.frame_mtx
            );

            if (!cam2.frame.empty()) {
                f2 = cam2.frame.clone();
            }
        }

        if (!f1.empty()) {
            cv::imshow("Cam1", f1);
        }

        if (!f2.empty()) {
            cv::imshow("Cam2", f2);
        }

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    running = false;

    t1.join();
    t2.join();

    return 0;
}
