#pragma once

#include <string>
#include <atomic>
#include "CameraStream.hpp"

extern std::atomic<bool> running;

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata);

void streamCamera(const std::string& url, CameraStream& stream);