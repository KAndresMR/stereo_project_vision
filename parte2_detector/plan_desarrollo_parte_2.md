# Plan de Desarrollo - Parte II (Proyecto Integrador)

A continuación se presenta un plan estructurado por fases, subfases y etapas sugeridas para abordar el desarrollo completo de la segunda parte del proyecto, garantizando que se cumplan todos los criterios de la rúbrica de evaluación.

---

## Fase 1: Construcción y Curación del Dataset
*El objetivo de esta fase es preparar los datos necesarios para entrenar el modelo clásico de reconocimiento de patrones.*

* **Etapa 1.1: Definición del Vehículo Objetivo.** 
  * Seleccionar el tipo de vehículo del ecosistema ecuatoriano (ej. bus azul, taxi amarillo, camioneta de flete).
  * Registrar/Verificar que ningún otro grupo tenga el mismo objetivo.
* **Etapa 1.2: Recolección de Imágenes Positivas.**
  * Capturar u obtener al menos **4.000 imágenes** del vehículo desde múltiples ángulos, con diferentes condiciones de iluminación.
* **Etapa 1.3: Recolección de Imágenes Negativas.**
  * Capturar u obtener al menos **4.000 imágenes** de fondos donde no exista el vehículo objetivo (calles vacías, otros tipos de vehículos, peatones, etc.).
* **Etapa 1.4: Aumento de Datos (Data Augmentation).**
  * Escribir un pequeño script en Python usando `Albumentations` u OpenCV para multiplicar las imágenes aplicando rotaciones leves, cambios de brillo, contraste, etc., y asegurar la robustez del modelo.

---

## Fase 2: Entrenamiento del Detector Clásico
*En esta fase entrenaremos el modelo matemático que correrá luego en C++.*

* **Etapa 2.1: Elección de la Técnica de Extracción.**
  * Decidir qué técnica utilizar: **HOG + SVM** o **LBP + Cascadas de Haar**. *(Nota: Usualmente las Cascadas de Haar con `opencv_traincascade` son más rápidas de implementar para video en tiempo real).*
* **Etapa 2.2: Entrenamiento del Modelo.**
  * Extraer las características y ejecutar el entrenamiento para obtener los pesos finales (ej. archivo `.xml` o modelo SVM guardado).
* **Etapa 2.3: Validación Temprana.**
  * Extraer las métricas del entrenamiento: armar la **Matriz de Confusión**, y calcular Precisión, Sensibilidad (Recall) y Especificidad para usarlas después en el informe.

---

## Fase 3: Aplicación "Disparador" en Escritorio (OpenCV C++)
*Desarrollo del componente en C++ que actuará como trigger en vivo.*

* **Etapa 3.1: Configuración de la Captura de Video.**
  * Instanciar `cv::VideoCapture` para leer en tiempo real desde la cámara.
  * Añadir impresión en consola del consumo de **RAM** y de los **Cuadros Por Segundo (FPS)**.
* **Etapa 3.2: Detección en Tiempo Real.**
  * Cargar el modelo entrenado en la Fase 2 y evaluar *frame a frame*.
  * Trazar el *bounding box* sobre el vehículo cuando la confianza de detección supere un umbral seguro, e imprimir dicho nivel de confianza (confidence).
* **Etapa 3.3: Lógica de Almacenamiento.**
  * Cuando se detecte el vehículo, guardar en memoria o disco el **frame clave**.
  * Empezar a grabar los siguientes frames para generar el **clip de video de 5 segundos**.
* **Etapa 3.4: Comunicación API (Cliente HTTP).**
  * Implementar una librería como `cURL` (o equivalente en C++) para enviar mediante peticiones HTTP (POST) la imagen y el clip de video a la API del Bot (Python).

---

## Fase 4: Segmentación Profunda y Bot de Telegram (Python)
*Desarrollo del receptor que utiliza Deep Learning para inferencia y segmentación.*

* **Etapa 4.1: Configuración del Bot de Telegram.**
  * Crear el bot con *BotFather* en Telegram, obtener el token y armar el script receptor usando librerías como `python-telegram-bot`, `telebot` o `FastAPI/Flask` (para recibir las peticiones de C++).
* **Etapa 4.2: Integración de YOLO (Ultralytics).**
  * Cargar un modelo YOLO pre-entrenado de segmentación (ej. `yolov8n-seg.pt` o `yolov11n-seg.pt`).
* **Etapa 4.3: Procesamiento de Imágenes y Video.**
  * Programar la lógica para que, al recibir un frame y un video de la API de C++, YOLO lo procese en modo inferencia y dibuje las máscaras de color translúcido sobre todos los vehículos y objetos de la escena.
* **Etapa 4.4: Envío de Respuesta al Usuario.**
  * El Bot envía por el chat de Telegram:
    1. Un mensaje de alerta y la **Imagen Original**.
    2. La **Imagen Segmentada**.
    3. El **Vídeo de 5 segundos** procesado y segmentado (como `.mp4` o GIF animado).

---

## Fase 5: Integración, Pruebas de Estrés y Optimización
*Sincronización de los dos componentes y pruebas finales.*

* **Etapa 5.1: Flujo End-to-End.**
  * Ejecutar ambas aplicaciones simultáneamente en la misma o en diferentes máquinas y probar el flujo ininterrumpido (Cámara -> C++ -> HTTP -> Python -> YOLO -> Telegram).
* **Etapa 5.2: Mitigación de Falsos Positivos.**
  * Realizar ajustes al umbral de confianza en C++ frente a diferentes ambientes.
* **Etapa 5.3: Pruebas de Estrés con Videos Reales.**
  * Cargar en la aplicación en C++ videos en alta calidad de calles ecuatorianas reales (en lugar de la cámara web) para simular la prueba en vivo requerida por el docente.

---

## Fase 6: Documentación y Video-Blog (Inglés)
*Preparación de los entregables textuales y audiovisuales.*

* **Etapa 6.1: Diagramas de Arquitectura.**
  * Dibujar el flujo de arquitectura de red (C++ comunicándose con la API del bot de Python).
* **Etapa 6.2: Consolidación de Métricas.**
  * Recopilar matrices de confusión, latencias de API, RAM, FPS para la presentación.
* **Etapa 6.3: Grabación del Video-Blog.**
  * Redactar guion en inglés y grabar la explicación del flujo y resultados cualitativos de las máscaras generadas por YOLO.
