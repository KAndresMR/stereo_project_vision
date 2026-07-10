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

// -----------------------------------------------------------------------------
// 1. UTILIDAD: Medir Consumo de RAM en macOS
// -----------------------------------------------------------------------------
double getMemoryUsageMB() {
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
    if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count)) {
        return -1.0;
    }
    return static_cast<double>(t_info.resident_size) / (1024.0 * 1024.0);
}

// -----------------------------------------------------------------------------
// 2. COMUNICACIÓN: Enviar datos vía HTTP POST (cURL)
// -----------------------------------------------------------------------------
void sendAlertToBot(const std::string& imagePath, const std::string& videoPath) {
    std::cout << "[HTTP] Preparando envío a la API del Bot...\n";
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

        // Endpoint del servidor Python (Fase 4)
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:5000/trigger");
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "[HTTP] ❌ Error enviando datos: " << curl_easy_strerror(res) << "\n";
        } else {
            std::cout << "[HTTP] ✅ Alerta enviada con éxito.\n";
        }

        curl_mime_free(form);
        curl_easy_cleanup(curl);
    }
}

// -----------------------------------------------------------------------------
// 3. ESTRUCTURAS DE DATOS COMPARTIDOS (Grabación Multihilo)
// -----------------------------------------------------------------------------
std::mutex qMutex;
std::queue<cv::Mat> videoQueue;
std::atomic<bool> isRecording(false);
std::atomic<bool> stopApp(false);

// Hilo que graba 5 segundos y envía a la API, para no congelar la cámara en vivo
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

            std::cout << ">>> [REC] Clip guardado con éxito. Ejecutando Etapa 3.4 (Envío HTTP)...\n";
            sendAlertToBot("evidencia_foto.jpg", videoFilename);
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
    std::cout << " INICIANDO APLICACIÓN TRIGGER (FASE 3) \n";
    std::cout << "=============================================\n\n";

    // 1. Etapa 3.2: Cargar el modelo matemático de HOG+SVM
    std::vector<float> svm_weights;
    std::ifstream file("../svm_hog_weights.txt");
    if (!file.is_open()) {
        std::cerr << "❌ [Error] No se encontró el archivo ../svm_hog_weights.txt generado en la Fase 2.\n";
        return -1;
    }
    
    float val;
    while (file >> val) {
        svm_weights.push_back(val);
    }
    file.close();

    // Recrear la misma configuración de ventana que usamos en Python
    cv::HOGDescriptor hog(cv::Size(64, 64), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
    hog.setSVMDetector(svm_weights);
    std::cout << "✅ [SVM] Modelo cargado con éxito (" << svm_weights.size() << " pesos).\n";

    // 2. Etapa 3.1: Iniciar Cámara Web de la PC
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "❌ [Error] No se pudo acceder a la cámara web (índice 0).\n";
        return -1;
    }
    
    // Configurar resolución estándar para evitar lag y RAM excesiva
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    int currentFps = cap.get(cv::CAP_PROP_FPS);
    if (currentFps <= 0) currentFps = 30; // Fallback si la cámara no reporta FPS
    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 3. Etapa 3.3: Iniciar Hilo de Grabación de Evidencias
    std::thread recorderThread(recorderThreadFunc, currentFps, frameSize);

    // 4. Bucle Principal (Evaluación en tiempo real)
    cv::Mat frame, gray;
    double tickFreq = cv::getTickFrequency();
    
    std::cout << "✅ [SISTEMA] Sistema en línea. Presiona 'q' en la ventana para salir.\n\n";

    while (true) {
        int64 t_start = cv::getTickCount();
        
        cap >> frame;
        if (frame.empty()) break;

        // Si estamos grabando, empujamos el frame limpio (sin letras) a la cola
        if (isRecording) {
            std::lock_guard<std::mutex> lock(qMutex);
            videoQueue.push(frame.clone());
        }

        // HOG requiere escala de grises
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> found_locations;
        std::vector<double> found_weights;
        
        // =========================================================================================
        // 🎛️ ZONA DE TUNING (AJUSTA ESTOS PARÁMETROS PARA ELIMINAR FALSOS POSITIVOS)
        // =========================================================================================
        
        // 1. PARAMETROS DE BARRIDO HOG
        double hit_threshold = 1.0;  // Distancia al hiperplano SVM. Súbelo a 2.0 o 3.0 para ser más estricto.
        cv::Size win_stride(16, 16); // Salto de píxeles. Valores comunes: (8,8), (16,16), (32,32). Mayor salto = menos falsos positivos y más rápido.
        cv::Size padding(16, 16);    // Padding de la ventana. Déjalo en (16,16) o prueba (8,8).
        double scale = 1.15;         // Escala de la pirámide (1.05 = minucioso, 1.15 = normal, 1.25 = rápido). Mayor valor = menos falsos positivos.

        // Ejecutamos el barrido en TODA la imagen (frame a frame) como dice la rúbrica
        hog.detectMultiScale(gray, found_locations, found_weights, hit_threshold, win_stride, padding, scale, 2.0, false);

        // 2. PARAMETROS DE LIMPIEZA (NMS - Non-Maximum Suppression)
        float nms_confidence_threshold = 3.5f; // Confianza mínima absoluta. Todo lo menor a esto se borra. Súbelo a 4.5 o 5.5 si sigues viendo basura.
        float nms_overlap_threshold = 0.3f;    // Qué tanto se permite que dos cajas se toquen (0.3 = 30%).
        
        // =========================================================================================

        // 🚀 Aplicar filtro NMS para dejar un solo Bounding Box
        std::vector<float> found_weights_float(found_weights.begin(), found_weights.end());
        std::vector<int> indices;
        cv::dnn::NMSBoxes(found_locations, found_weights_float, nms_confidence_threshold, nms_overlap_threshold, indices);

        for (int idx : indices) {
            cv::Rect r = found_locations[idx];
            double confidence = found_weights[idx];

            // Dibujar caja delimitadora (Bounding Box)
            cv::rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
            
            std::string label = cv::format("Furgoneta (Conf: %.2f)", confidence);
            cv::putText(frame, label, cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            // Lógica de "Trigger"
            if (!isRecording) {
                std::cout << "\n🚨 ¡FURGONETA DETECTADA! Nivel de confianza: " << confidence << "\n";
                std::cout << "   Guardando fotografía clave...\n";
                cv::imwrite("evidencia_foto.jpg", frame);
                
                // Al poner esto en true, el hilo paralelo empieza a grabar 5 segundos de inmediato
                isRecording = true; 
            }
        }

        // Etapa 3.1: Imprimir telemetría en pantalla
        int64 t_end = cv::getTickCount();
        double fps = tickFreq / (t_end - t_start);
        double ramMB = getMemoryUsageMB();

        std::string stats = cv::format("FPS: %.1f | Consumo RAM: %.1f MB", fps, ramMB);
        cv::putText(frame, stats, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // Mostrar el resultado en vivo
        cv::imshow("Deteccion HOG+SVM (Fase 3)", frame);

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
