# 🚌 Sistema de Detección de Buses Escolares — HOG+SVM & YOLOv8-seg

**Proyecto Integrador (Parte II) — Visión Artificial**  
**Arquitectura:** HOG + SVM (OpenCV C++) → API HTTP → YOLOv8-seg + Bot de Telegram (Python)  
**Plataforma de Desarrollo:** macOS (Apple Silicon M1/M2/M3)

---

## 1. Descripción Ejecutiva

Este proyecto implementa un sistema híbrido de visión artificial en tiempo real para la detección y notificación de buses escolares:
1. **Motor de Detección en En Vivo (C++):** Captura el flujo de video web, procesa fotogramas mediante descriptores geométricos **HOG (Histogram of Oriented Gradients)** y clasifica con un modelo **SVM Lineal**.
2. **Trigger y Concurrencia:** Al confirmar una detección persistente, un hilo asíncrono graba 5 segundos de video y transmite evidencias (fotografía y clip) mediante HTTP POST (libcurl).
3. **Segmentación de Instancias y Notificación (Python):** Un servidor Flask recibe las evidencias, segmenta los objetos en escena frame a frame utilizando **YOLOv8 Nano (Segmentación)** y notifica al usuario en Telegram con las imágenes y el video procesados.

---

## 2. Objeto de Estudio: Bus Escolar Clásico

Se seleccionó el **bus escolar estadounidense clásico** (*Glossy Yellow*) por sus características geométricas altamente distintivas para el descriptor HOG:
- **Trompa frontal prominente:** Genera un gradiente diagonal y ortogonal único frente al parabrisas.
- **Patrones ortogonales definidos:** Líneas horizontales marcadas (techo, laterales) y repetición vertical constante de los marcos de las ventanas.
- **Relación de aspecto (Aspect Ratio):** El análisis del dataset demostró una relación promedio de **1.30 (ancho/alto)**, motivo por el cual se descartaron furgonetas planas que confundían al clasificador y se fijó una ventana estándar de **128×96 píxeles**.

---

## 3. Construcción y Curación del Dataset

Para garantizar la generalización y cumplir con los requisitos de robustez, se consolidó un dataset de **8,315 imágenes perfectamente balanceadas**:

### 3.1 Imágenes Positivas (~4,000 muestras)
- **Recolección:** Se extrajeron recortes (crops) automatizados con YOLO desde 4 repositorios de *Roboflow Universe* y muestras filtradas desde *Google Open Images V7*.
- **Curación con Inteligencia Artificial:** Se pasó un pre-filtro con **ResNet50** (clase 779 de ImageNet = *School Bus*) para descartar automáticamente ambulancias, furgonetas de carga y autobuses urbanos, complementado con inspección visual.
- **Aumento de Datos (Albumentations):** Se aplicaron transformaciones geométricas y de color que respetan la física de los gradientes (Flip Horizontal, ajustes leves de Brillo/Contraste y Rotaciones ±5°).  
  > *Nota técnica:* Se descartó explícitamente el aumento por **Ruido Gaussiano**, ya que inyecta miles de gradientes falsos en todas las direcciones, destruyendo la silueta que HOG necesita aprender.

### 3.2 Imágenes Negativas (~4,400 muestras)
- **Fondo genérico:** Carreteras, vehículos comunes (sedanes, SUV), peatones y paisajes urbanos.
- **Fondo de prueba real (Hard Negatives del entorno):** Se extrajeron 147 fotogramas del entorno de pruebas (habitación, muebles, cortinas, paredes). Incluir el fondo real enseña al SVM a suprimir falsos positivos causados por pliegues verticales o patrones locales en la habitación.

---

## 4. Pipeline de Preprocesamiento y Entrenamiento Machine Learning

Para evitar discrepancias en inferencia (*features mismatch*), **el pipeline matemático es 100% idéntico en Python y en C++**:

1. **Escala de Grises:** HOG evalúa variaciones espaciales de intensidad (bordes); se eliminan los canales RGB para reducir la carga computacional al 33% sin pérdida de precisión.
2. **Ecualización Adaptativa (CLAHE):** Se aplica `clipLimit=2.0` con bloques de `8×8` para resaltar bordes en zonas de sombra (chasis y llantas negras).
3. **Redimensionamiento con Padding:** Las imágenes se escalan al tamaño **128×96** rellenando bordes con negro puro (el negro uniforme tiene gradiente cero y es invisible para HOG, evitando deformar la geometría del vehículo).

### 4.1 Entrenamiento y Exportación del Modelo SVM
- **Descriptor HOG:** Configurado con bloques de 16x16, celdas de 8x8, saltos de 8x8 y 9 orientaciones de histograma, generando un vector de **5,940 características** por frame.
- **Clasificador:** Entrenado con `LinearSVC` (Scikit-Learn) bajo búsqueda de hiperparámetros (Grid Search), alcanzando un **Accuracy del 89.96%** con C=0.1.
- **Exportación para OpenCV C++:** El vector de pesos del hiperplano (w) y el término independiente (b) se unifican en un archivo de **5,941 coeficientes** (`svm_hog_weights_128x96.txt`). Se validó matemáticamente el signo positivo del intercepto (+b) para coincidir con el producto punto nativo de OpenCV.

---

## 5. Arquitectura del Sistema y Flujo End-to-End

```
+--------------------+        HTTP POST (cURL)       +----------------------+
|  C++ TriggerApp    | ----------------------------> | Python Flask Server  |
|  (OpenCV HOG+SVM)  |    evidencia_foto.jpg         | (telegram_server.py) |
|  Cámara en vivo    |    evidencia_video.mp4        | Inferencia YOLOv8-seg|
+--------------------+                               +----------------------+
                                                                |
                                                                | Telegram API (POST)
                                                                v
                                                     +----------------------+
                                                     |   Bot de Telegram    |
                                                     |  1. Alerta + Original|
                                                     |  2. Foto Segmentada  |
                                                     |  3. Video Segmentado |
                                                     +----------------------+
```

1. **Detección C++:** Analiza video a 640×480. Aplica supresión de no-máximos (**NMS**) para unificar cajas delimitadoras.
2. **Confirmación y Grabación:** Al detectar al vehículo objetivo continuamente, activa una alarma temporal, guarda el frame clave y despliega un hilo secundario (`std::thread`) que captura exactamente 5 segundos de video en H.264 (AVC1).
3. **Transmisión y Limpieza:** Envía los archivos por HTTP POST y los borra automáticamente del disco local (`std::remove`).
4. **Segmentación YOLOv8:** El servidor procesa la imagen y aplica inferencia frame a frame sobre el MP4, superponiendo máscaras translúcidas de segmentación de instancias.
5. **Entrega de Resultados:** Envia a Telegram el reporte formal estructurado en tres evidencias.

---

## 6. Estructura Limpia y Modular del Proyecto

```
parte2_detector/
├── CMakeLists.txt              # Configuración general de compilación C++
├── README.md                   # Documentación técnica general (este archivo)
├── X_features.npy              # Array numpy de características HOG extraídas
├── y_labels.npy                # Array numpy de etiquetas de clase
├── svm_hog_weights_128x96.txt  # 5,941 coeficientes exportados para OpenCV C++
├── build/                      # Directorio de construcción y ejecutable TriggerApp
├── dataset/                    # Repositorio de imágenes balanceado
│   ├── positives/              # ~4,000 imágenes de buses escolares curadas
│   └── negatives/              # ~4,400 imágenes de fondo y entorno
├── notebooks/                  # Cuadernos de experimentación Jupyter
│   ├── Dataset.ipynb           # Extracción, CLAHE y aumento Albumentations
│   └── Reentrenamiento.ipynb   # Entrenamiento LinearSVC y métricas
├── src/                        # Código fuente C++ (OpenCV)
│   └── main.cpp                # Sistema principal multihilo de detección
├── telegram_bot/               # Sub-sistema receptor y notificador
│   ├── telegram_server.py      # Servidor Flask e inferencia YOLO
│   └── yolov8n-seg.pt          # Pesos del modelo de segmentación
└── videos/                     # Videos de prueba y registros de control
    ├── cuarto.mov              
    └── video.mov               
```

---

## 7. Guía Rápida de Ejecución y Pruebas

### Paso 1: Iniciar el Servidor de Telegram + YOLO (Terminal 1)
Desde la raíz del proyecto (`parte2_detector/`), inicia el servidor receptor:
```bash
python telegram_bot/telegram_server.py
```
> *El servidor escuchará en el puerto 5001. Las credenciales de Telegram se leen automáticamente por variables de entorno (`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`) o utilizan las credenciales configuradas por defecto.*

### Paso 2: Compilar y Ejecutar la Aplicación en C++ (Terminal 2)
Inicia el sistema de detección visual en tiempo real:
```bash
cd build
cmake .. && make
./TriggerApp
```

### 🎛️ Recomendaciones para Pruebas en Vivo (Controles de la Interfaz)
Una vez abierta la ventana `Detector HOG+SVM`, podrás calibrar la detección en tiempo real utilizando los controles deslizantes (*Trackbars*):
- **Hit Threshold (x10):** Umbral de decisión del SVM. Valor recomendado: `5` a `10` (representa 0.5 a 1.0 en escala SVM). Disminúyelo si la iluminación es baja y no detecta el vehículo; auméntalo si observas falsas detecciones.
- **NMS Conf (x10):** Confianza mínima de filtrado de cajas superpuestas. Valor estándar: `10` (1.0).
- **Scale (x100):** Factor de escala de la pirámide HOG. Valor óptimo para cámara web: `115` (1.15).

### 💡 Notas Operativas
- **Detener la aplicación:** Presiona la tecla **`q`** directamente sobre la ventana de OpenCV para realizar un apagado limpio de hilos y cámara.
- **Gestión de Memoria y Residuos:** Tanto la app en C++ como el servidor Python gestionan sus evidencias en memoria temporal y eliminan los archivos residuales tras cada transmisión, manteniendo el sistema limpio.
- **Monitoreo:** Puedes observar los FPS en vivo y el consumo residente de memoria RAM directamente en la esquina superior izquierda de la pantalla de video.
