#pragma once
#include <opencv2/opencv.hpp>

class Kalman1D {
public:
    // processNoise: Qué tan rápido crees que se mueve el objeto real (más alto = reacciona más rápido, pero filtra menos).
    // measurementNoise: Cuánto confías en la cámara vs la matemática (más alto = confía menos en la cámara, suaviza más).
    Kalman1D(float processNoise = 1e-3, float measurementNoise = 0.1) {
        // 1 variable de estado (Z), 1 medida (Z), 0 variables de control
        kf_.init(1, 1, 0);

        // Matriz de transición (estado actual = estado anterior)
        kf_.transitionMatrix = (cv::Mat_<float>(1, 1) << 1);

        // Matriz de medida (medimos directamente Z)
        cv::setIdentity(kf_.measurementMatrix);

        // Matrices de ruido
        cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(processNoise));
        cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(measurementNoise));
        cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

        initialized_ = false;
    }

    float update(float measurement) {
        // Si la medición es inválida (ej. el objeto salió de cámara), devolvemos el último estado conocido
        if (measurement <= 0.0f) {
            if (!initialized_) return 0.0f;
            return kf_.statePost.at<float>(0);
        }

        // Si es la primera medición válida, saltamos la matemática y fijamos el estado inicial
        if (!initialized_) {
            kf_.statePost.at<float>(0) = measurement;
            initialized_ = true;
            return measurement;
        }

        // 1. Predicción (¿Dónde debería estar el objeto?)
        kf_.predict();

        // 2. Corrección (Ajustar la predicción con lo que realmente vio la cámara)
        cv::Mat_<float> measurementMat(1, 1);
        measurementMat(0) = measurement;
        cv::Mat estimated = kf_.correct(measurementMat);

        return estimated.at<float>(0);
    }

private:
    cv::KalmanFilter kf_;
    bool initialized_;
};