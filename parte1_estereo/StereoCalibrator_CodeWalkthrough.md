# 🔭 Guía Completa: `StereoCalibrator.cpp`

> 🔗 **Flujo:** [Paso Anterior → MonoCalibrator](file:///Users/andresmorocho/Downloads/Vision/practicas/stereo_project/MonoCalibrator_CodeWalkthrough.md) ➔ **Estás aquí (Calibración Estéreo)** ➔ *El sistema está listo para medir profundidad en tiempo real.*

---

## ¿Por qué existe este archivo?

Ahora que `MonoCalibrator` ya sabe cómo "ve" cada cámara por separado (cuál es su foco, dónde está su centro óptico, cómo distorsiona el lente), necesitamos resolver una pregunta diferente:

> **"¿Cómo están físicamente ubicadas la cámara izquierda y la derecha entre sí?"**

Piénsalo así: tienes dos observadores (tus dos cámaras) mirando al mismo objeto desde ángulos levemente diferentes. Para medir la distancia a ese objeto usando geometría (como hacen tus ojos biológicamente), necesitas saber **exactamente** cuánto están separados esos dos observadores y si uno está más alto o más rotado que el otro.

`StereoCalibrator.cpp` descubre esa relación espacial exacta entre tus dos cámaras ESP32.

---

## 🧮 La Matemática que Necesitas Entender Primero

### La Geometría Epipolar

Cuando una cámara izquierda y una derecha miran al mismo punto P en el espacio:

```
                    P (objeto en el mundo)
                   /|
                  / |
                 /  |
          ------/----------- 
         |  L  /   |   R   |
         | cam/    |  cam  |     L = Cámara Izquierda
          ---/-----|---------     R = Cámara Derecha
            /      |
           uL      uR
     (proyección  (proyección
      en imagen    en imagen
      izquierda)   derecha)
```

La línea que conecta `uL` (donde se ve en la imagen izquierda) con `uR` (donde se ve en la imagen derecha) es la **Línea Epipolar**. La distancia horizontal entre `uL` y `uR` se llama **Disparidad** y es inversamente proporcional a la profundidad Z.

Para que esto funcione, necesitamos encontrar dos cosas:
1. **R (Matriz de Rotación 3×3):** ¿Está una cámara rotada respecto a la otra?
2. **T (Vector de Traslación 3×1):** ¿Cuánto están separadas en X, Y y Z?

### La Fórmula de Profundidad (La Clave de Todo)

Una vez que sabes R y T, la profundidad de cualquier objeto se calcula con:

```
         fx × B
Z = ────────────
           d

Donde:
  fx = Distancia focal horizontal (en píxeles, ~614px de tu OV2640)
  B  = Baseline = distancia entre cámaras (en metros, ~0.075m en tu setup)
  d  = Disparidad = diferencia horizontal en píxeles entre L y R
  Z  = Profundidad en metros
```

**Ejemplo con tus valores reales:**
```
Tu cámara tiene fx=614.76 px y B=0.07531 m
Si un objeto aparece con d=50px de disparidad:

Z = (614.76 × 0.07531) / 50
Z = 46.29 / 50
Z = 0.9258 metros ≈ 92 centímetros
```

---

## 🔬 Análisis Función por Función

### Función 1: `buildObjectPoints()` — Reutilizando el Patrón

```cpp
// Código real de tu StereoCalibrator.cpp (líneas 18-27)
std::vector<cv::Point3f> StereoCalibrator::buildObjectPoints() const {
    std::vector<cv::Point3f> pts;
    pts.reserve(config_.boardSize.width * config_.boardSize.height);
    for (int r = 0; r < config_.boardSize.height; ++r)
        for (int c = 0; c < config_.boardSize.width; ++c)
            pts.push_back({c * config_.squareSizeM,
                           r * config_.squareSizeM,
                           0.0f});
    return pts;
}
```

Esta función es idéntica a la del MonoCalibrator. La cuadrícula 3D del tablero es la misma independientemente de si estás calibrando la cámara izquierda o derecha, o el par estéreo. El tablero siempre mide lo mismo físicamente.

---

### Función 2: `loadIntrinsics()` — El Puente entre Fases

**¿Qué hace?** Lee los archivos YAML que generó MonoCalibrator y carga las matrices K y D de ambas cámaras en memoria RAM.

```cpp
// Código real de tu StereoCalibrator.cpp (líneas 35-65)
bool StereoCalibrator::loadIntrinsics(cv::Mat& K_left,  cv::Mat& dist_left,
                                      cv::Mat& K_right, cv::Mat& dist_right,
                                      cv::Size& imageSize) const {
    // Lambda que abre un YAML y extrae los datos
    auto load = [&](const std::string& path,
                    cv::Mat& K, cv::Mat& dist,
                    const std::string& name) -> bool {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            Log::error("StereoCalib", "Cannot open: " + path);
            Log::error("StereoCalib", "Run mono calibration first (menu option 3).");
            return false;
        }
        fs["camera_matrix"]           >> K;     // Lee la matriz K del YAML
        fs["distortion_coefficients"] >> dist;  // Lee los coeficientes D del YAML
        int w = 0, h = 0;
        fs["image_width"]  >> w;
        fs["image_height"] >> h;
        if (w > 0 && h > 0) imageSize = {w, h};
        fs.release();
        Log::yamlLoaded(path, true);
        return true;
    };

    // Carga ambos YAMLs usando la lambda
    return load(config_.leftYaml,  K_left,  dist_left,  "LEFT ")
        && load(config_.rightYaml, K_right, dist_right, "RIGHT");
}
```

**¿Qué es esa sintaxis `auto load = [&](...) -> bool { ... }`?**

Es una función Lambda de C++11. Es como definir una función pequeña dentro de otra función, sin tener que darle un nombre global. La `[&]` significa que puede acceder a todas las variables locales de la función que la contiene (como `config_` y `Log`).

**¿Por qué el error dice "Run mono calibration first"?**
Porque si los YAML no existen, es 100% seguro que el usuario se saltó el paso anterior. Tu código anticipa este error y da una instrucción clara en lugar de crashing con un puntero nulo.

**¿Qué diferencia a `>>` de `<<` en FileStorage?**
```cpp
fs << "clave" << valor;   // WRITE: guarda 'valor' bajo la etiqueta 'clave'
fs["clave"]   >> variable; // READ: carga el valor de 'clave' en 'variable'
```

---

### Función 3: `detectPair()` — Exigencia de Pareja Completa

**¿Qué hace?** Abre el par de imágenes `left_XX.jpg` y `right_XX.jpg`, y solo devuelve `true` si AMBAS cámaras encontraron el tablero completo.

```cpp
// Código real de tu StereoCalibrator.cpp (líneas 75-129)
bool StereoCalibrator::detectPair(const std::string& leftPath,
                                  const std::string& rightPath,
                                  std::vector<cv::Point2f>& cornersL,
                                  std::vector<cv::Point2f>& cornersR,
                                  cv::Size& imageSize) const {
    cv::Mat imgL = cv::imread(leftPath,  cv::IMREAD_COLOR);
    cv::Mat imgR = cv::imread(rightPath, cv::IMREAD_COLOR);

    if (imgL.empty() || imgR.empty()) return false;

    if (imgL.size() != imgR.size()) {
        Log::warn("StereoCalib", "Image size mismatch between LEFT and RIGHT.");
        return false;
    }

    cv::Mat grayL, grayR;
    cv::cvtColor(imgL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, grayR, cv::COLOR_BGR2GRAY);

    cv::Mat dummyL = imgL, dummyR = imgR;
    DetectionResult dL = detector_.detect(grayL, dummyL);
    DetectionResult dR = detector_.detect(grayR, dummyR);

    if (dL.found && dR.found) {    // ← CONDICIÓN BINARIA: ambas o ninguna
        cornersL = dL.corners;
        cornersR = dR.corners;
        Log::info("StereoCalib", "  ✓ " + fname + " ...");
        return true;
    }

    // Explica cuál de las dos cámaras falló
    std::string reason;
    if (!dL.found && !dR.found) reason = "BOTH cameras failed";
    else if (!dL.found)         reason = "LEFT failed (corners=" + ...;
    else                         reason = "RIGHT failed (corners=" + ...;

    Log::warn("StereoCalib", "  ✗ " + fname + " | " + reason + " → SKIPPED");
    return false;
}
```

**¿Por qué la condición es `&&` (ambas) y no `||` (al menos una)?**

La matemática de `stereoCalibrate` necesita **correspondencia punto a punto**. Eso significa que `cornersL[0]` (la esquina superior izquierda detectada en la imagen izquierda) debe corresponder con `cornersR[0]` (esa misma esquina física detectada en la imagen derecha). Si la cámara derecha solo encuentra 53 de 54 esquinas, no sabemos cuál de las 54 falta, y por lo tanto no podemos hacer la correspondencia. El par completo es inútil.

**Analogía:** Es como hacer traducción paralela. Si tienes el libro en español con 300 páginas y el mismo libro en inglés con solo 297 páginas (algunas se perdieron), no puedes hacer un diccionario de traducciones precisas sin saber cuáles páginas faltan.

---

### Función 4: `calibrate()` — El Corazón Matemático del Proyecto

**¿Qué hace?** Ejecuta `cv::stereoCalibrate` con todos los pares de esquinas para encontrar R y T.

```cpp
// Código real de tu StereoCalibrator.cpp (líneas 185-286)
StereoCalibrator::Result StereoCalibrator::calibrate() const {

    // PASO 1: Cargar intrínsecos de la fase anterior
    cv::Mat K_left, dist_left, K_right, dist_right;
    cv::Size imageSize;
    if (!loadIntrinsics(K_left, dist_left, K_right, dist_right, imageSize))
        return result;

    // PASO 2: Cargar todos los pares de imágenes
    std::vector<std::string> leftPaths, rightPaths;
    cv::glob(config_.datasetDir + "/left/*.jpg",  leftPaths,  false);
    cv::glob(config_.datasetDir + "/right/*.jpg", rightPaths, false);
    std::sort(leftPaths.begin(),  leftPaths.end());
    std::sort(rightPaths.begin(), rightPaths.end());

    // PASO 3: Para cada par, extraer esquinas de ambas cámaras
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imgPtsL, imgPtsR;
    for (int i = 0; i < result.pairsTotal; ++i) {
        std::vector<cv::Point2f> cL, cR;
        if (detectPair(leftPaths[i], rightPaths[i], cL, cR, sz)) {
            objectPoints.push_back(singleObjPts);
            imgPtsL.push_back(cL);
            imgPtsR.push_back(cR);
        }
    }

    // PASO 4: Ejecutar la calibración estéreo
    try {
        result.rpe = cv::stereoCalibrate(
            objectPoints,          // La cuadrícula 3D (misma para todas las fotos)
            imgPtsL, imgPtsR,      // Esquinas detectadas en L y R para cada foto
            K_left,  dist_left,    // Intrínsecos de la cámara izquierda
            K_right, dist_right,   // Intrínsecos de la cámara derecha
            imageSize,
            result.R, result.T,   // ← SALIDAS: Rotación y Traslación
            result.E, result.F,   // ← SALIDAS: Matriz Esencial y Fundamental
            cv::CALIB_FIX_INTRINSIC,  // ← NO tocar K ni D, solo buscar R y T
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                100, 1e-6)         // Máximo 100 iteraciones o error < 0.000001
        );
    } catch (const cv::Exception& e) {
        Log::error("StereoCalib", "stereoCalibrate threw: " + ...);
        return result;
    }

    // PASO 5: Calcular el Baseline (distancia física entre cámaras)
    result.baselineM = cv::norm(result.T);  // = sqrt(Tx² + Ty² + Tz²)
}
```

**El flag `cv::CALIB_FIX_INTRINSIC` — la decisión más importante del archivo:**

```
Sin el flag (modo libre):
  OpenCV busca: K_left, D_left, K_right, D_right, R, T
  Total de variables a optimizar: 4×4 + 2×5 + 3 + 3 = 32 variables
  → El optimizador tiene demasiados grados de libertad.
    Puede "compensar" errores del dataset modificando las focales,
    generando matrices K incorrectas.

Con CALIB_FIX_INTRINSIC (tu elección):
  OpenCV busca: SOLO R, T
  Total de variables a optimizar: 3 + 3 = 6 variables
  → El optimizador tiene un espacio de búsqueda pequeño.
    Levenberg-Marquardt converge rápido y de forma confiable.
    Las focales (ya calibradas por MonoCalibrator) no se tocan.
```

**¿Qué son E y F (Matriz Esencial y Fundamental)?**

```
E = Essential Matrix  → Relación entre puntos en coordenadas normalizadas (independiente del lente)
F = Fundamental Matrix → Relación entre píxeles reales en las dos imágenes

Matemáticamente: E = K_right^T × F × K_left

Tu código las calcula pero no las usa directamente para nada más.
OpenCV las genera automáticamente como parte de stereoCalibrate.
Son útiles si quisieras hacer verificación epipolar o rectificación manual.
```

**Calculando el Baseline:**
```cpp
result.baselineM = cv::norm(result.T);
```

`cv::norm` calcula la norma Euclidiana: $\|T\| = \sqrt{T_x^2 + T_y^2 + T_z^2}$

Si tu T es `[-0.0753, 0.0012, 0.0008]` (metros), entonces:
```
Baseline = sqrt(0.0753² + 0.0012² + 0.0008²)
         = sqrt(0.005670 + 0.0000014 + 0.00000064)
         = sqrt(0.005672)
         = 0.07531 metros = 75.31 mm
```
Ese 75.31mm es la distancia física medida matemáticamente entre tus dos cámaras.

---

### Función 5: `validateResult()` — Auditoría de Hardware

```cpp
// Código real de tu StereoCalibrator.cpp (líneas 134-180)
void StereoCalibrator::validateResult(const Result& result) const {

    // ── Verificar Baseline ───────────────────────────────────────────────
    double blMm = result.baselineM * 1000.0;  // Convertir a mm para leer mejor
    Log::info(tag, "Baseline: " + std::to_string(blMm) + " mm");
    if (result.baselineM < 0.03)
        Log::warn(tag, "Baseline < 30mm — cámaras muy juntas o T está mal");
    else if (result.baselineM > 0.20)
        Log::warn(tag, "Baseline > 200mm — inusualmente grande");

    // ── Verificar dirección del vector T ───────────────────────────────
    double tx = std::abs(result.T.at<double>(0));  // Componente horizontal
    double ty = std::abs(result.T.at<double>(1));  // Componente vertical
    double tz = std::abs(result.T.at<double>(2));  // Componente profundidad
    if (ty > tx * 0.3 || tz > tx * 0.3)
        Log::warn(tag, "T tiene componente vertical/profundidad significativa");
    else
        Log::info(tag, "T direction OK (primarily horizontal) ✓");

    // ── Verificar ángulo de rotación entre cámaras ─────────────────────
    cv::Mat rvec;
    cv::Rodrigues(result.R, rvec);                    // Matriz 3x3 → vector 3x1
    double angleRad = cv::norm(rvec);                  // Norma del vector = ángulo en radianes
    double angleDeg = angleRad * 180.0 / CV_PI;       // Convertir a grados
    Log::info(tag, "Rotation angle: " + std::to_string(angleDeg) + " deg");
    if (angleDeg > 5.0)
        Log::warn(tag, "Rotation > 5° — desalineación física significativa");
}
```

**Entendiendo la verificación de T:**

Si tus cámaras están una al lado de la otra horizontalmente (que es tu caso), el vector T debería verse así:
```
T = [-0.0753,   0.0012,   0.0008]
      ↑            ↑          ↑
  Tx=-75mm    Ty=1.2mm   Tz=0.8mm
  (Baseline)  (casi 0)   (casi 0)
```

La verificación `ty > tx * 0.3` significa: "¿Es el desplazamiento vertical más del 30% del horizontal?" Si Tx=75mm y Ty > 22.5mm, significa que una cámara está montada significativamente más alta que la otra, lo que causaría problemas en el mapa de disparidad.

**El Algoritmo de Rodrigues — Convirtiendo una Matriz a un Ángulo:**

La Matriz de Rotación R es una grilla 3×3 de 9 números. No puedes leer directamente cuántos grados está rotada. El algoritmo de Rodrigues la convierte a un vector 3D donde:
- La **dirección** del vector = eje alrededor del cual ocurre la rotación
- La **magnitud** (longitud) del vector = ángulo de rotación en radianes

```
R  (3×3 incomprensible)
        ↓  cv::Rodrigues()
rvec = [rx, ry, rz]  (vector de 3 números)
        ↓  cv::norm()
ángulo_rad = sqrt(rx² + ry² + rz²)
        ↓  × 180/π
ángulo_deg = 2.3°   ← legible y útil
```

---

### Función 6: `saveYAML()` — Guardando Todo + Rectificación

Este es el acto final y más complejo. No solo guarda R y T, sino que también calcula y guarda los mapas de rectificación que el modo de disparidad usará en tiempo real.

```cpp
// Código real de tu StereoCalibrator.cpp (líneas 294-367)
bool StereoCalibrator::saveYAML(const Result& result) const {

    // PASO 1: Recalcular la rectificación (necesita los intrínsecos de nuevo)
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(
        K_left,  dist_left,
        K_right, dist_right,
        imageSize,
        result.R, result.T,    // Los R y T que acabamos de calcular
        R1, R2, P1, P2, Q,    // ← SALIDAS
        cv::CALIB_ZERO_DISPARITY,  // Alinear centros ópticos horizontalmente
        0,                          // alpha=0: recortar bordes negros
        imageSize
    );

    // PASO 2: Escribir TODO al YAML (lo que stereoCalibrate calculó + lo que stereoRectify calculó)
    fs << "R"  << result.R   // Rotación entre cámaras
       << "T"  << result.T   // Traslación entre cámaras (contiene el Baseline)
       << "E"  << result.E   // Matriz Esencial
       << "F"  << result.F   // Matriz Fundamental
       << "R1" << R1         // Rotación de rectificación para cámara izquierda
       << "R2" << R2         // Rotación de rectificación para cámara derecha
       << "P1" << P1         // Proyección rectificada izquierda
       << "P2" << P2         // Proyección rectificada derecha
       << "Q"  << Q;         // ← LA JOYA: Disparity-to-Depth Reprojection Matrix
}
```

**¿Qué hace `cv::stereoRectify`?**

`stereoCalibrate` te dice *cómo están* las cámaras. `stereoRectify` te dice *cómo deformar artificialmente las imágenes* para que ambas cámaras parezcan perfectamente alineadas horizontalmente.

```
Antes de rectificar:            Después de rectificar:
┌─────────────┐                 ┌─────────────┐
│ CAM IZQUIERDA│                │ CAM IZQUIERDA│
│  ●           │                │         ●   │
│  (objeto)    │                │  (objeto)   │
└─────────────┘                 └─────────────┘
┌─────────────┐                 ┌─────────────┐
│  CAM DERECHA │                │  CAM DERECHA │
│    ●         │ ──rectify──>   │         ●   │
│   (objeto,   │                │  (objeto,   │
│  pero en     │                │  misma Y    │
│  otra Y)     │                │  que L)     │
└─────────────┘                 └─────────────┘
```
Después de rectificar, si el objeto aparece en la fila Y=150 de la cámara izquierda, también aparecerá exactamente en la fila Y=150 de la cámara derecha. Esto hace que la búsqueda de disparidad sea un problema 1D (solo buscar en la misma fila) en lugar de 2D.

**`alpha=0` — El recorte necesario:**

Al rotar una imagen digitalmente, las esquinas quedan vacías (píxeles negros). El parámetro `alpha=0` le dice a OpenCV que haga zoom-in hasta que desaparezcan todos los píxeles negros. Resultado: imagen más pequeña, pero sin bordes negros.

```
alpha=1 (conserva todo)        alpha=0 (recorta a válido)
┌─────────────────────┐        ┌─────────────┐
│ ███████             │        │             │
│ ████████████████    │ ──>    │  ████████   │
│ █████████████████   │        │  ████████   │
│ ████████████████    │        │  ████████   │
│              ███    │        │             │
└─────────────────────┘        └─────────────┘
```

**La Matriz Q — Convirtiendo Disparidad en Profundidad:**

`Q` es la razón de ser de todo este proceso. Es una matriz 4×4 que encapsula la relación entre un píxel con su disparidad y la distancia en el mundo real.

```
Tu Matriz Q real (ejemplo):
Q = [ 1   0    0    -cx_L  ]    ← cx_L = 320 (centro óptico izquierdo)
    [ 0   1    0    -cy    ]    ← cy   = 240 (centro óptico vertical)
    [ 0   0    0     fx    ]    ← fx   = 614.76 (distancia focal)
    [ 0   0  -1/B  (cx_L-cx_R)/B ]  ← B = 0.07531m (Baseline)

Para usar: si un píxel está en (u=320, v=240) con disparidad d=50px:
  [X]   [u - cx_L  ]   [0    ]
  [Y] = Q × [v - cy    ] = [0    ]  → X=0, Y=0, Z=0.925m (92.5cm)
  [Z]   [d         ]   [0.925]
  [W]   [1         ]   [...]
```

Tu `DisparityMode.cpp` **no usa la multiplicación directa de Q** (eso es para nubes de puntos 3D densas). En cambio, usa la fórmula simplificada directamente:
```cpp
z = kalmanFijo.update((FOCAL_PX * BASELINE_M / disp) * 100.0f);
// Donde: FOCAL_PX = 614.76f y BASELINE_M = 0.07531f
```
Que es exactamente la misma fórmula del centro de Q: `Z = fx × B / d`.

---

## 🧪 Preguntas de Comprensión

### Nivel Básico

**P1:** ¿Qué pasa si `loadIntrinsics()` no puede abrir `left_calib.yml`?
> **R:** Retorna `false` e imprime "Run mono calibration first". La función `calibrate()` recibe ese `false` y aborta sin ejecutar `stereoCalibrate`.

**P2:** Si tienes 30 pares de imágenes pero solo 24 tienen el tablero visible en ambas cámaras, ¿cuántos pares usa `stereoCalibrate`?
> **R:** 24 pares. `detectPair()` filtra los 6 donde alguna cámara falló. El resultado mostrará: `Valid stereo pairs: 24 / 30`.

**P3:** El Baseline de tu setup es ~75mm. ¿Qué significa un objeto con disparidad d=100px? (fx=614)
> **R:** `Z = 614.76 × 0.07531 / 100 = 0.4629m ≈ 46 cm`. Está muy cerca.

### Nivel Intermedio

**P4:** ¿Por qué `saveYAML()` necesita volver a llamar a `loadIntrinsics()` si ya se llamó antes en `calibrate()`?
> **R:** Porque `saveYAML()` es un método separado que no recibe las matrices K y D como parámetros. La arquitectura modular las mantiene encapsuladas, así que `saveYAML()` las recarga desde YAML cuando las necesita para llamar a `stereoRectify`.

**P5:** Si las cámaras están montadas perfectamente, ¿qué valor debería tener el ángulo que calcula Rodrigues?
> **R:** Exactamente 0°. En la práctica, por tolerancias de impresión 3D, la estructura siempre tiene algún pequeño ángulo. Un ángulo menor a 1° es excelente; menor a 5° es aceptable.

### Nivel Avanzado (Para Jurado)

**P6:** Tu código usa `TermCriteria(EPS + MAX_ITER, 100, 1e-6)`. ¿Qué significa cada parte?
> **R:** `EPS + MAX_ITER` significa "detén cuando CUALQUIERA de los dos criterios se cumpla". `100` es el número máximo de iteraciones del algoritmo de Levenberg-Marquardt. `1e-6` es el épsilon: si la mejora entre una iteración y la siguiente es menor a `0.000001`, el algoritmo considera que convergió y se detiene, aunque no haya llegado a las 100 iteraciones.

**P7:** ¿Por qué el código verifica que `imgL.size() != imgR.size()` en `detectPair()` si ya verificamos resoluciones en MonoCalibrator?
> **R:** Defensa en profundidad. MonoCalibrator garantiza que todas las fotos de `left/` son 640×480 y todas las de `right/` son 640×480. Pero `detectPair()` compara la izquierda con la derecha de ese par específico. Si una sesión de captura tuvo problemas (reinicio de cámara a mitad), podría pasar que `left_15.jpg` sea 640×480 y `right_15.jpg` tenga una resolución diferente por ser de una sesión distinta. Esta verificación lo atrapa.
