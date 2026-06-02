#include "StreamWorker.hpp"
#include <curl/curl.h>
#include <thread>
#include <chrono>
#include <algorithm>

const uchar JPEG_SOI[] = {0xFF, 0xD8};
const uchar JPEG_EOI[] = {0xFF, 0xD9};

std::atomic<bool> running(true);

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    if (!running) return 0;

    auto* stream = static_cast<CameraStream*>(userdata);
    size_t total = size * nmemb;

    {
        std::lock_guard<std::mutex> lock(stream->buffer_mtx);
        stream->buffer.insert(
            stream->buffer.end(),
            ptr,
            ptr + total
        );
    }

    return total;
}

void streamCamera(const std::string& url, CameraStream& stream) {
    CURL* curl = curl_easy_init();
    if (!curl) return;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    std::thread processThread([&]() {
        while (running) {
            std::vector<uchar> localBuffer;
            {
                std::lock_guard<std::mutex> lock(stream.buffer_mtx);
                localBuffer = stream.buffer;
            }

            auto soi = std::search(
                localBuffer.begin(),
                localBuffer.end(),
                JPEG_SOI,
                JPEG_SOI + 2
            );

            if (soi == localBuffer.end()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            auto eoi = std::search(
                soi + 2,
                localBuffer.end(),
                JPEG_EOI,
                JPEG_EOI + 2
            );

            if (eoi == localBuffer.end()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            std::vector<uchar> jpg(soi, eoi + 2);
            cv::Mat img = cv::imdecode(jpg, cv::IMREAD_COLOR);

            if (!img.empty()) {
                auto now = std::chrono::steady_clock::now();
                {
                    std::lock_guard<std::mutex> lock(stream.frame_mtx);
                    stream.frame = img;
                    stream.timestamp = now;
                }

                stream.frameCount++;
                double elapsed = std::chrono::duration<double>(
                    now - stream.lastFpsTime
                ).count();

                if (elapsed >= 1.0) {
                    stream.fps = stream.frameCount / elapsed;
                    stream.frameCount = 0;
                    stream.lastFpsTime = now;
                }
            }

            size_t consumed = (eoi - localBuffer.begin()) + 2;
            {
                std::lock_guard<std::mutex> lock(stream.buffer_mtx);
                if (consumed <= stream.buffer.size()) {
                    stream.buffer.erase(
                        stream.buffer.begin(),
                        stream.buffer.begin() + consumed
                    );
                }
            }
        }
    });

    curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    processThread.join();
}