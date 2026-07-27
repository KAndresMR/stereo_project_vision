#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <chrono>
#include <curl/curl.h>
#include <mach/mach.h> // Librería nativa de macOS para medir RAM (Apple Silicon)
#include <cstdio>

// ---------------------------------------------------------------------------
// Medición de memoria residente del proceso (API Mach de macOS).
// Devuelve el consumo actual en megabytes.
// ---------------------------------------------------------------------------
double getMemoryUsageMB() {
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
    if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count)) {
        return -1.0;
    }
    return static_cast<double>(t_info.resident_size) / (1024.0 * 1024.0);
}

// ---------------------------------------------------------------------------
// Envío de evidencias (imagen y video) vía HTTP POST utilizando libcurl.
// ---------------------------------------------------------------------------
void sendAlertToBot(const std::string& imagePath, const std::string& videoPath) {
    std::cout << "[HTTP] Preparando envío al servidor...\n";
    CURL* curl = curl_easy_init();
    if (curl) {
        curl_mime* form = curl_mime_init(curl);
        curl_mimepart* field;

        // Adjuntar imagen clave
        field = curl_mime_addpart(form);
        curl_mime_name(field, "photo");
        curl_mime_filedata(field, imagePath.c_str());

        // Adjuntar clip de video
        field = curl_mime_addpart(form);
        curl_mime_name(field, "video");
        curl_mime_filedata(field, videoPath.c_str());

        // Endpoint del servidor Python de recepción y segmentación
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:5001/trigger");
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "[HTTP] [ERROR] Falló el envío de datos: " << curl_easy_strerror(res) << "\n";
        } else {
            std::cout << "[HTTP] [OK] Alerta enviada con éxito.\n";
        }

        curl_mime_free(form);
        curl_easy_cleanup(curl);
    }
}

// ---------------------------------------------------------------------------
// Estructuras de sincronización para concurrencia (captura y grabación)
// ---------------------------------------------------------------------------
std::mutex qMutex;
std::queue<cv::Mat> videoQueue;
std::atomic<bool> isRecording(false);
std::atomic<bool> stopApp(false);

// Parámetros dinámicos vinculados a la interfaz gráfica (trackbars)
int hit_thresh_slider = 5;      // Representa 0.5 (x10)
int nms_conf_slider = 10;       // Representa 1.0 (x10)
int scale_slider = 115;         // Representa 1.15 (x100)

// Función ejecutada en hilo secundario para grabar clips de evidencia y enviarlos
void recorderThreadFunc(int fps, cv::Size size) {
    while (!stopApp) {
        if (isRecording) {
            std::string videoFilename = "evidencia_video.mp4";
            
            // Codec H.264 (avc1) compatible con Telegram
            cv::VideoWriter writer(videoFilename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, size);
            
            if (!writer.isOpened()) {
                std::cerr << "[Error] No se pudo abrir VideoWriter para guardar el clip.\n";
                isRecording = false;
                continue;
            }

            int framesToRecord = fps * 5; // Exactamente 5 segundos de grabación
            int framesRecorded = 0;

            std::cout << ">>> [REC] Iniciando captura de clip (" << framesToRecord << " frames)...\n";

            while (framesRecorded < framesToRecord && !stopApp) {
                cv::Mat frame;
                {
                    std::lock_guard<std::mutex> lock(qMutex);
                    if (!videoQueue.empty()) {
                        frame = videoQueue.front();
                        videoQueue.pop();
                    }
                }

                if (!frame.empty()) {
                    writer.write(frame);
                    framesRecorded++;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            
            writer.release();
            isRecording = false; // Liberar el candado
            
            // Vaciar la cola de frames sobrantes
            {
                std::lock_guard<std::mutex> lock(qMutex);
                while (!videoQueue.empty()) videoQueue.pop();
            }

            std::cout << ">>> [REC] Clip guardado con éxito. Iniciando transmisión HTTP...\n";
            sendAlertToBot("evidencia_foto.jpg", videoFilename);
            std::remove("evidencia_foto.jpg");
            std::remove(videoFilename.c_str());
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main() {
    std::cout << "\n=============================================\n";
    std::cout << " INICIANDO SISTEMA DE DETECCIÓN EN VIVO \n";
    std::cout << "=============================================\n\n";

    // Cargar el modelo clasificador HOG+SVM entrenado previamente
    std::vector<float> svm_weights;
    std::ifstream file("../svm_hog_weights_128x96.txt");
    if (!file.is_open()) {
        std::cerr << "[ERROR] No se encontró el archivo de pesos ../svm_hog_weights_128x96.txt\n";
        return -1;
    }
    
    float val;
    while (file >> val) {
        svm_weights.push_back(val);
    }
    file.close();

    // Configuración del descriptor HOG con las mismas dimensiones del entrenamiento
    cv::HOGDescriptor hog(cv::Size(128, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
    hog.setSVMDetector(svm_weights);
    std::cout << "[OK] Modelo SVM cargado en memoria (" << svm_weights.size() << " coeficientes).\n";

    // Inicialización del dispositivo de captura de video
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] No se pudo acceder a la cámara web (índice 0).\n";
        return -1;
    }
    
    // Configurar resolución estándar para evitar lag y RAM excesiva
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    int currentFps = cap.get(cv::CAP_PROP_FPS);
    if (currentFps <= 0) currentFps = 30; // Fallback si la cámara no reporta FPS
    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Inicio del hilo asíncrono para gestión de evidencias de video
    std::thread recorderThread(recorderThreadFunc, currentFps, frameSize);

    // Bucle principal de evaluación en tiempo real
    cv::Mat frame, gray;
    double tickFreq = cv::getTickFrequency();
    
    // Configuración de la ventana y barra de controles dinámicos
    const std::string winName = "Detector HOG+SVM";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Hit Threshold (x10)", winName, &hit_thresh_slider, 30);
    cv::createTrackbar("NMS Conf (x10)", winName, &nms_conf_slider, 30);
    cv::createTrackbar("Scale (x100)", winName, &scale_slider, 150);

    // Inicializar CLAHE una sola vez fuera del bucle para evitar asignaciones repetidas en memoria
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    std::cout << "[INFO] Sistema en línea. Presiona 'q' en la ventana para salir.\n\n";

    while (true) {
        int64 t_start = cv::getTickCount();
        
        cap >> frame;
        if (frame.empty()) break;

        // Almacenar frame limpio en la cola de grabación si el proceso está activo
        if (isRecording) {
            std::lock_guard<std::mutex> lock(qMutex);
            videoQueue.push(frame.clone());
        }

        // Conversión a escala de grises para procesamiento del descriptor HOG
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Aplicación de ecualización adaptativa de histograma (CLAHE)
        clahe->apply(gray, gray);

        std::vector<cv::Rect> found_locations;
        std::vector<double> found_weights;
        
        // Lectura de parámetros de calibración en tiempo real desde los controles
        if (scale_slider <= 100) scale_slider = 101; // Previene bucle infinito por factor de escala <= 1.0
        
        double hit_threshold = static_cast<double>(hit_thresh_slider) / 10.0;
        double scale = static_cast<double>(scale_slider) / 100.0;
        cv::Size win_stride(16, 16); 
        cv::Size padding(16, 16);    

        // Detección multiescala en la imagen rectificada.
        // Se establece finalThreshold = 0.0 para conservar las confianzas sin agrupar por defecto.
        hog.detectMultiScale(gray, found_locations, found_weights, hit_threshold, win_stride, padding, scale, 0.0, false);

        float nms_confidence_threshold = static_cast<float>(nms_conf_slider) / 10.0f;
        float nms_overlap_threshold = 0.3f;    

        // Supresión de no-máximos (NMS) para unificar detecciones solapadas en la ventana con mayor confianza
        std::vector<float> found_weights_float(found_weights.begin(), found_weights.end());
        std::vector<int> indices;
        cv::dnn::NMSBoxes(found_locations, found_weights_float, nms_confidence_threshold, nms_overlap_threshold, indices);

        // Lógica de "Confirmación de 2 segundos"
        static int detectionFrameCount = 0;
        const int REQUIRED_FRAMES = 60; // Aproximadamente 2 segundos a 30 FPS

        if (!indices.empty()) {
            detectionFrameCount++;
            
            // Dibujamos solo la mejor caja (la primera en indices suele ser la de mayor confianza por el NMS)
            int idx = indices[0];
            cv::Rect r = found_locations[idx];
            double confidence = found_weights[idx];

            // Dibujar caja delimitadora (Bounding Box)
            cv::rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
            
            std::string label = cv::format("Furgoneta (Conf: %.2f)", confidence);
            cv::putText(frame, label, cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            // Mostrar progreso de confirmación
            if (!isRecording && detectionFrameCount < REQUIRED_FRAMES) {
                std::string progressMsg = cv::format("Confirmando... %d / %d frames", detectionFrameCount, REQUIRED_FRAMES);
                cv::putText(frame, progressMsg, cv::Point(r.x, r.y - 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
            }

            // Activación del evento al superar el umbral de persistencia temporal
            if (!isRecording && detectionFrameCount >= REQUIRED_FRAMES) {
                std::cout << "\n[ALERTA] ¡Vehículo objetivo confirmado! Nivel de confianza: " << confidence << "\n";
                std::cout << "         Guardando fotografía clave de evidencia...\n";
                cv::imwrite("evidencia_foto.jpg", frame);
                
                // Al poner esto en true, el hilo paralelo empieza a grabar 5 segundos de inmediato
                isRecording = true;
                
                // Reseteamos el contador para la próxima vez (después de que termine de grabar)
                detectionFrameCount = 0; 
            }
        } else {
            // Si en este frame NO hay detección, reseteamos el contador (exige detecciones consecutivas)
            detectionFrameCount = 0;
        }

        // Cálculo y visualización de telemetría de rendimiento (FPS y memoria RAM)
        int64 t_end = cv::getTickCount();
        double fps = tickFreq / (t_end - t_start);
        double ramMB = getMemoryUsageMB();

        std::string stats = cv::format("FPS: %.1f | Consumo RAM: %.1f MB", fps, ramMB);
        cv::putText(frame, stats, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // Actualizar visualización principal
        cv::imshow("Detector HOG+SVM", frame);

        // Salir con 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Limpieza al salir
    std::cout << "[SISTEMA] Apagando aplicación...\n";
    stopApp = true;
    recorderThread.join();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
