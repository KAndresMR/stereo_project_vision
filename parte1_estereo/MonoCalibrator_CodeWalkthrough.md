# 📷 Guía Completa: `MonoCalibrator.cpp`

> 🔗 **Flujo:** [Paso Anterior → Captura de Dataset](file:///Users/andresmorocho/.gemini/antigravity-ide/brain/cb723f05-ad4e-4e36-b9b8-ac45bdf6f7ef/README_OPCION_1_DATASET.md) ➔ **Estás aquí (Calibración Monocular)** ➔ [Siguiente → StereoCalibrator](file:///Users/andresmorocho/Downloads/Vision/practicas/stereo_project/StereoCalibrator_CodeWalkthrough.md)

---

## ¿Por qué existe este archivo?

Imagina que acabas de comprarte unos anteojos nuevos. Aunque los cristales son casi perfectos, tienen pequeñísimas imperfecciones que hacen que las líneas rectas parezcan levemente curvas en los bordes. Un optometrista mide esas imperfecciones y las codifica en tu receta.

**`MonoCalibrator.cpp` es el optometrista de tu cámara ESP32.** Analiza tus 30 fotos del tablero de ajedrez y descubre exactamente cuánto y cómo el lente deforma la imagen. Hace esto por **cada cámara por separado** (izquierda, luego derecha).

---

## 🧮 La Matemática que Necesitas Entender Primero

Antes de leer el código, necesitas saber qué está calculando matemáticamente.

### El Modelo Pinhole (Cámara Estenopeica)

Una cámara digital es fundamentalmente un **agujero con un sensor**. Todo punto del mundo real pasa por ese agujero y aterriza en el sensor como un píxel. La ecuación que describe esto es:

```
Un punto 3D del mundo real  →  un píxel 2D en la imagen
     (X, Y, Z)             →       (u, v)
```

La fórmula exacta que lo convierte es la **Matriz K** (también llamada *Matriz Intrínseca*):

```
         | fx   0   cx |
K   =    |  0  fy   cy |
         |  0   0    1 |
```

**¿Qué significa cada número?**

| Símbolo | Nombre           | Valor típico OV2640 | Qué es en el mundo físico                     |
|---------|------------------|---------------------|-----------------------------------------------|
| `fx`    | Foco horizontal  | ~614 px             | Si mueves algo 1m horizontalmente, ¿cuántos píxeles se desplaza? |
| `fy`    | Foco vertical    | ~613 px             | Lo mismo pero verticalmente                   |
| `cx`    | Centro óptico X  | ~320 px             | El píxel del centro horizontal de la imagen   |
| `cy`    | Centro óptico Y  | ~240 px             | El píxel del centro vertical de la imagen     |

> 💡 **Analogía simple:** `fx` y `fy` son como el zoom de un telescopio. Un valor alto (800px) significa mucho zoom. Un valor bajo (200px) significa gran angular. Tu OV2640 tiene un gran angular moderado, por eso sus focales rondan los 600px.

### La Distorsión de Barril

El lente esférico no es perfecto. Curva las líneas rectas. Tu código calcula 5 números que describen esta curvatura:

```
D = [k1, k2, p1, p2, k3]
```

La corrección matemática que se aplica a cada píxel es:

```
Para un punto detectado en (x_dist, y_dist):
r² = x_dist² + y_dist²

x_correcto = x_dist × (1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x·y + p2·(r²+2x²)
y_correcto = y_dist × (1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r²+2y²) + 2·p2·x·y
```

**¿Qué significa esto visualmente?**

```
Imagen con distorsión de barril    →    Imagen corregida
(lente del OV2640 sin calibrar)         (después de aplicar D)

        ┌─────────┐                          ┌─────────┐
       /  ╔═════╗  \                         │ ╔═════╗ │
      /   ║  ■  ║   \        k1<0            │ ║  ■  ║ │
     /    ║     ║    \  ══════════════>       │ ║     ║ │
      \   ╚═════╝   /                        │ ╚═════╝ │
       \            /                         └─────────┘
        └──────────┘
    (bordes curvados hacia afuera)      (cuadrado perfecto)
```

`k1` negativo = Distorsión de barril (OV2640 típico: `-0.3` a `-0.6`)  
`k1` positivo = Distorsión de cojín (NO es lo que tienes)

---

## 🔬 Análisis Función por Función

### Función 1: `buildObjectPoints()`

**¿Qué hace?** Crea una lista de coordenadas 3D que representan el tablero de ajedrez **ideal y perfecto**, sin ninguna deformación de lente.

```cpp
// Código real de tu MonoCalibrator.cpp (líneas 47-61)
std::vector<cv::Point3f> MonoCalibrator::buildObjectPoints() const {
    std::vector<cv::Point3f> pts;
    pts.reserve(config_.boardSize.width * config_.boardSize.height);

    for (int row = 0; row < config_.boardSize.height; ++row) {
        for (int col = 0; col < config_.boardSize.width; ++col) {
            pts.push_back({
                col * config_.squareSizeM,   // X en metros
                row * config_.squareSizeM,   // Y en metros
                0.0f                          // Z = SIEMPRE CERO
            });
        }
    }
    return pts;
}
```

**¿Qué genera exactamente este código con tu tablero 9x6?**

Para un tablero 9×6 con cuadros de 25mm (`squareSizeM = 0.025f`), los primeros puntos son:

```
(0.000, 0.000, 0.0)   ← esquina superior izquierda
(0.025, 0.000, 0.0)   ← siguiente esquina, 25mm a la derecha
(0.050, 0.000, 0.0)
(0.075, 0.000, 0.0)
...
(0.200, 0.000, 0.0)   ← fin de la primera fila (8 columnas × 0.025 = 0.200m)
(0.000, 0.025, 0.0)   ← inicio de la segunda fila
...
(0.200, 0.125, 0.0)   ← última esquina (fila 6 × 0.025 = 0.125m)
```

**¿Por qué `Z = 0.0f`?**

El tablero es una hoja de papel plana. Si algo es plano, todas sus coordenadas Z son cero. Esta es la razón técnica por la que el algoritmo funciona: cuando Z=0, la matemática de proyección se simplifica drásticamente y OpenCV puede resolver el sistema de ecuaciones analíticamente.

**¿Por qué usar metros?**  
Un comentario de tu propio código lo dice claramente (línea 44-45):
```
// These are in METERS because squareSizeM is in meters. This makes the
// translation vector T from stereoCalibrate come out in meters too — so
// later, Z = f*B/d gives depth in meters directly.
```
Si usas milímetros aquí, el Baseline (B) que calculará StereoCalibrator también saldría en milímetros, y tu fórmula de profundidad `Z = f*B/d` daría profundidades en milímetros. Al usar metros aquí, **todo el pipeline de profundidad queda automáticamente en metros** sin conversiones extra.

**`pts.reserve(...)`:** Reserva memoria anticipadamente para los 54 puntos. Sin esto, el vector de C++ se va redimensionando dinámicamente cada vez que agregar un punto (lento). Con reserve, asigna todo el bloque de memoria de una sola vez (rápido).

---

### Función 2: `loadAndDetect()`

**¿Qué hace?** Abre una foto del disco, la convierte a escala de grises, y le pide a `ChessboardDetector` que encuentre las 54 esquinas del tablero con precisión sub-píxel.

```cpp
// Código real de tu MonoCalibrator.cpp (líneas 73-114)
bool MonoCalibrator::loadAndDetect(const std::string& imagePath,
                                   std::vector<cv::Point2f>& outCorners,
                                   cv::Size& outImageSize) const {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        Log::error("MonoCalib", "Cannot read: " + imagePath);
        return false;
    }

    outImageSize = img.size();   // Guarda el tamaño: 640x480

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);  // Color → Grises

    cv::Mat displayDummy = img;   // Copia para dibujar (no se guarda)
    DetectionResult det  = detector_.detect(gray, displayDummy);

    if (det.found) {
        Log::info("MonoCalib", "  ✓ " + fname + " | brightness=" + ...);
    } else {
        Log::warn("MonoCalib", "  ✗ " + fname + " → SKIPPED");
    }

    if (!det.found) return false;

    outCorners = det.corners;  // Las 54 esquinas encontradas con sub-píxel
    return true;
}
```

**¿Por qué convertir a escala de grises?**
`findChessboardCorners` detecta esquinas buscando cambios bruscos de intensidad (del blanco al negro). En escala de grises, ese cambio es un solo número (de 255 a 0). En color, serían 3 números (RGB). Operar en escala de grises reduce los datos a procesar en 3x, haciendo la detección 3 veces más rápida sin pérdida de precisión para esta tarea.

**¿Qué es `displayDummy`?**
`detector_.detect()` dibuja las esquinas encontradas directamente sobre la imagen que le pases. Aquí se le pasa una **copia** del frame (`displayDummy = img`), no el original. Esto es crucial: si le pasamos el original, las líneas dibujadas encima de las esquinas contaminarían la foto que luego se intentaría usar para calibrar.

**¿Qué contiene `det.corners` cuando tiene éxito?**
Es un vector de 54 coordenadas en píxeles con decimales:
```
[ (47.23, 38.91), (93.77, 38.14), (140.52, 37.66), ... ]
```
Nota los decimales: esto es el resultado del refinamiento sub-píxel (`cornerSubPix`) que hizo `ChessboardDetector`. Sin ese refinamiento, serían coordenadas enteras: `(47, 38)`, `(93, 38)`, lo que genera un error de calibración mucho mayor.

---

### Función 3: `calibrate()` — El Corazón del Sistema

**¿Qué hace?** Orquesta todo: encuentra las imágenes, extrae las esquinas de cada una, y se las pasa a OpenCV para que haga la optimización matemática.

```cpp
// Código real de tu MonoCalibrator.cpp (líneas 184-308)
MonoCalibrator::Result MonoCalibrator::calibrate(Side side) const {

    // ── PASO 1: Encuentra todas las fotos en el disco ──────────────────
    std::vector<std::string> imagePaths;
    cv::glob(dir + "/*.jpg", imagePaths, false);
    std::sort(imagePaths.begin(), imagePaths.end());  // orden predecible

    // ── PASO 2: Para cada foto, extraer esquinas ───────────────────────
    std::vector<std::vector<cv::Point3f>> objectPoints;  // La cuadrícula 3D perfecta
    std::vector<std::vector<cv::Point2f>> imagePoints;   // Los píxeles 2D detectados

    const std::vector<cv::Point3f> singleObjPts = buildObjectPoints();

    for (const auto& path : imagePaths) {
        std::vector<cv::Point2f> corners;
        cv::Size sz;

        if (loadAndDetect(path, corners, sz)) {
            if (imageSize.empty()) imageSize = sz;
            if (sz != imageSize) continue;  // Imagen de resolución diferente → saltar

            objectPoints.push_back(singleObjPts);  // La cuadrícula ideal
            imagePoints.push_back(corners);          // Las esquinas detectadas
        }
    }

    // ── PASO 3: Ejecutar la optimización ──────────────────────────────
    std::vector<cv::Mat> rvecs, tvecs;
    try {
        result.rpe = cv::calibrateCamera(
            objectPoints,     // Lo que DEBERÍA verse (3D ideal)
            imagePoints,      // Lo que REALMENTE se ve (2D detectado)
            imageSize,        // Tamaño de la imagen: 640x480
            result.cameraMatrix,   // ← SALIDA: La Matriz K
            result.distCoeffs,     // ← SALIDA: Los coeficientes D
            rvecs,
            tvecs
        );
    } catch (const cv::Exception& e) {
        Log::error("MonoCalib", "calibrateCamera threw: " + std::string(e.what()));
        return result;
    }
}
```

**Entendiendo el loop de datos:**

```
Para la foto left_01.jpg:
  objectPoints[0] = [(0,0,0), (0.025,0,0), ...]  ← SIEMPRE IGUAL (la cuadrícula perfecta)
  imagePoints[0]  = [(47.2, 38.9), (93.7, 38.1), ...] ← Detectado en esa foto específica

Para la foto left_02.jpg:
  objectPoints[1] = [(0,0,0), (0.025,0,0), ...]  ← SIEMPRE IGUAL
  imagePoints[1]  = [(55.1, 42.3), (101.4, 41.8), ...] ← Detectado en esa foto (diferente ángulo)
```

**¿Qué hace `cv::calibrateCamera` matemáticamente?**

Le estamos diciendo: *"Mira, yo sé cómo debería verse el tablero (los objectPoints). Tú me dices cómo lo captó realmente la cámara (los imagePoints). Encuentra la Matriz K y los coeficientes D que minimicen la diferencia entre los dos".*

OpenCV ejecuta iteraciones del algoritmo de Levenberg-Marquardt hasta que la diferencia entre la proyección matemática y los píxeles reales sea mínima. El valor final de esa diferencia mínima se llama **RMS (Root Mean Square)** o **RPE (Reprojection Error)**.

```
Lo que dice tu código en el YAML resultante (ejemplo real):
  rms_reprojection_error: 0.3241

Interpretación:
  RMS < 0.3  → Excelente calibración
  RMS < 0.5  → Buena calibración
  RMS < 1.0  → Aceptable
  RMS > 1.0  → Recapturar imágenes
```

**¿Por qué verificar `sz != imageSize`?**
`cv::calibrateCamera` exige que TODAS las imágenes sean del mismo tamaño. Si accidentalmente capturaste una foto en 320x240 y el resto en 640x480, esa foto haría explotar el solver. Tu código detecta esto y simplemente saltea esa imagen en lugar de crashear.

---

### Función 4: `validateResult()` — El Verificador de Física Real

**¿Qué hace?** Una vez que OpenCV devuelve la matriz K y los coeficientes D, verifica que los números tengan sentido físico para un sensor OV2640 de 640x480 píxeles.

```cpp
// Código real de tu MonoCalibrator.cpp (líneas 130-179)
void MonoCalibrator::validateResult(const Result& result) const {
    const cv::Mat& K = result.cameraMatrix;
    const cv::Mat& D = result.distCoeffs;

    double fx = K.at<double>(0, 0);  // fila 0, columna 0
    double fy = K.at<double>(1, 1);  // fila 1, columna 1
    double cx = K.at<double>(0, 2);  // fila 0, columna 2
    double cy = K.at<double>(1, 2);  // fila 1, columna 2
    double k1 = D.at<double>(0);     // primer coeficiente de distorsión

    // Rangos esperados para OV2640 VGA 640x480:
    // fx, fy: 350-750 px
    // cx    : 270-370 px (debe estar cerca de 640/2 = 320)
    // cy    : 190-290 px (debe estar cerca de 480/2 = 240)
    // k1    : -0.8 a 0.1 (OV2640 típicamente -0.3 a -0.6)

    if (fx < 300 || fx > 900)
        Log::warn(tag, "fx fuera de rango — resultado sospechoso");

    if (std::abs(fx - fy) / std::max(fx, fy) > 0.05)
        Log::warn(tag, "fx y fy difieren > 5% — posible problema");

    if (k1 > 0.1)
        Log::warn(tag, "k1 positivo (efecto cojín). OV2640 normalmente es negativo (barril).");
    if (k1 < -0.9)
        Log::warn(tag, "k1 muy negativo — dataset probablemente corrupto.");

    if (result.rpe > 1.0)
        Log::warn(tag, "RMS > 1.0 px — recapturar imágenes.");

    if (result.rpe < 0.5 && result.imagesUsed >= 15)
        Log::info(tag, "Result looks good ✓");
}
```

**Leyendo la Matriz K posición por posición:**

```
La Matriz K es una grilla 3x3:

     columna 0   columna 1   columna 2
fila 0 [ fx=614      0        cx=320  ]
fila 1 [    0       fy=613    cy=240  ]
fila 2 [    0        0          1     ]

Entonces:
K.at<double>(0, 0) = 614   ← fx está en fila 0, columna 0
K.at<double>(1, 1) = 613   ← fy está en fila 1, columna 1
K.at<double>(0, 2) = 320   ← cx está en fila 0, columna 2
K.at<double>(1, 2) = 240   ← cy está en fila 1, columna 2
```

**¿Por qué k1 debe ser negativo?**
El lente del OV2640 es una lente convexa (abultada hacia afuera). Esto curva la luz hacia adentro, haciendo que los bordes de la imagen aparezcan "comprimidos" hacia el centro. En la fórmula matemática, ese efecto de compresión se expresa con un `k1` negativo. Si tu calibración arroja un k1 positivo, significa que el algoritmo está describiendo el efecto contrario (un lente cóncavo), lo que físicamente no tienes. Ese resultado sería inválido.

---

### Función 5: `saveYAML()` — La Caja Fuerte de los Resultados

**¿Qué hace?** Guarda la Matriz K y los coeficientes D en un archivo `.yml` que el siguiente paso (StereoCalibrator) podrá leer.

```cpp
// Código real de tu MonoCalibrator.cpp (líneas 313-350)
bool MonoCalibrator::saveYAML(const Result& result, Side side) const {
    if (!result.success) {
        Log::error("MonoCalib", "Cannot save YAML — calibration was not successful.");
        return false;     // No guarda un resultado fallido
    }

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        Log::error("MonoCalib", "Cannot open for writing: " + path);
        return false;
    }

    // Guarda metadatos para referencia humana
    fs << "camera_name"            << sideName(side)     // "LEFT" o "RIGHT"
       << "image_width"            << result.imageSize.width   // 640
       << "image_height"           << result.imageSize.height  // 480
       << "board_width"            << config_.boardSize.width  // 9
       << "board_height"           << config_.boardSize.height // 6
       << "square_size_m"          << config_.squareSizeM      // 0.025
       << "images_used"            << result.imagesUsed        // ej: 28
       << "images_total"           << result.imagesTotal       // ej: 30
       << "rms_reprojection_error" << result.rpe;              // ej: 0.324

    // Guarda las matrices matemáticas (la razón de ser del archivo)
    fs << "camera_matrix"           << result.cameraMatrix   // La Matriz K
       << "distortion_coefficients" << result.distCoeffs;    // El vector D

    fs.release();
    return true;
}
```

**¿Qué aspecto tiene el archivo `left_calib.yml` generado?**

```yaml
%YAML:1.0
camera_name: "LEFT"
image_width: 640
image_height: 480
board_width: 9
board_height: 6
square_size_m: 0.025
images_used: 28
images_total: 30
rms_reprojection_error: 0.3241

camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 6.1476e+02, 0., 3.1822e+02,
           0., 6.1401e+02, 2.4135e+02,
           0., 0., 1. ]

distortion_coefficients: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ -3.521e-01, 1.247e-01, 2.1e-04, -1.3e-04, -5.2e-02 ]
```

---

## 🧪 Preguntas de Comprensión

### Nivel Básico (Conceptual)

**P1:** Si tu tablero tiene cuadros de 30mm en lugar de 25mm, ¿en qué línea específica de `buildObjectPoints()` tienes que cambiar el valor?
> **R:** En la configuración, cambiando `squareSizeM = 0.030f`. El bucle usa `col * config_.squareSizeM`, así que automáticamente el cambio se propaga.

**P2:** ¿Por qué la función `loadAndDetect()` retorna `false` si no encuentra el tablero, en lugar de simplemente usar los datos parciales que encontró?
> **R:** Porque `cv::calibrateCamera` exige correspondencia perfecta: cada cuadrícula 3D del `objectPoints` debe tener exactamente 54 puntos 2D detectados en `imagePoints`. Si detectas 50 de 54 esquinas, no sabes cuáles 4 faltan ni dónde estarían en la cuadrícula 3D, haciendo imposible la correspondencia.

**P3:** ¿Qué significa un RMS de 0.32 px?
> **R:** Significa que si proyectas matemáticamente las esquinas 3D del tablero usando la Matriz K calculada, el promedio de diferencia entre donde *caería* cada punto proyectado y donde *realmente detectaste* la esquina en la foto es de 0.32 píxeles. Es excelente.

### Nivel Intermedio (Técnico)

**P4:** En el loop de `calibrate()`, ¿por qué `objectPoints` siempre recibe el mismo `singleObjPts` para cada foto, pero `imagePoints` siempre recibe datos diferentes?
> **R:** Porque el tablero físico siempre mide lo mismo (la cuadrícula 3D no cambia), pero la cámara lo fotografía desde ángulos distintos cada vez. En la foto 1 el tablero aparece centrado, en la foto 2 está rotado 30°, etc. Las coordenadas 2D en la imagen cambian con cada pose. La relación `objectPoints[i] ↔ imagePoints[i]` le dice al algoritmo: "Este arreglo 3D se veía así en la foto i".

**P5:** El código hace `if (sz != imageSize) continue;` dentro del loop. ¿Qué situación práctica podría generar este problema?
> **R:** Si reiniciaste la cámara durante la sesión de captura y esta cambió la resolución, o si mezclaste fotos de sesiones con diferentes configuraciones. También puede ocurrir si la función `cv::imread` devuelve un tamaño diferente por corrupción del archivo JPEG.

### Nivel Avanzado (Para Jurado)

**P6:** ¿Qué pasaría matemáticamente si tomaste todas las 30 fotos del tablero exactamente desde el mismo ángulo y distancia?
> **R:** El sistema de ecuaciones lineales que resuelve Levenberg-Marquardt se volvería **subdeterminado** o altamente redundante. Al tener poca varianza en las poses, la información para estimar los coeficientes de distorsión (especialmente `k2` y `k3`) sería insuficiente, ya que esos términos de orden superior solo se distinguen claramente cuando el tablero aparece en los bordes extremos de la imagen. El RMS podría verse bajo, pero la Matriz K sería imprecisa.

**P7:** ¿Por qué el código hace `std::sort(imagePaths)` antes de procesar las fotos?
> **R:** Para garantizar reproducibilidad. `cv::glob` no garantiza el orden del sistema de archivos (puede variar entre macOS y Linux). Si el orden es aleatorio, dos ejecuciones del mismo código sobre el mismo dataset podrían dar RMS ligeramente diferentes (por el orden en que el solver ve los datos). Con `sort`, siempre es `left_01, left_02, ...`, haciendo el resultado determinístico.
