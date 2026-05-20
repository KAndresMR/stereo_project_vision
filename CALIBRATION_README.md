# 📐 Módulo de Calibración Estéreo — Documentación Técnica Completa

> **Cómo leer este documento**
> Cada archivo tiene su propia sección con: qué problema resuelve, por qué existe separado,
> y cada método explicado línea por línea cuando importa. Los bloques 📝 son ejercicios de
> cuaderno. Los bloques 💡 son conceptos teóricos. Los bloques ⚠️ son decisiones de diseño
> importantes y por qué se tomaron así.

---

## Índice

1. [Arquitectura general — el mapa completo](#1-arquitectura-general)
2. [CalibrationConfig.hpp](#2-calibrationconfighpp)
3. [ChessboardDetector.hpp / .cpp](#3-chessboarddetector)
4. [CalibrationSession.hpp / .cpp](#4-calibrationsession)
5. [CalibrationMode.hpp / .cpp](#5-calibrationmode)
6. [main.cpp actualizado](#6-maincpp)
7. [CMakeLists.txt](#7-cmakeliststxt)
8. [Flujo completo de datos](#8-flujo-completo-de-datos)
9. [Cómo usar el sistema](#9-cómo-usar-el-sistema)
10. [Qué hacer con los pares capturados](#10-qué-sigue)

---

# 1. Arquitectura General

## El problema que resolvemos

Tenés dos cámaras transmitiendo video. El objetivo de esta fase es recolectar entre 20 y 30
pares de imágenes sincronizadas de un tablero de ajedrez, tomadas desde diferentes ángulos y
distancias. Esos pares son el insumo del algoritmo `stereoCalibrate` de OpenCV, que calculará
las matrices K, distorsión, R, T, E y F — todo lo que necesitás para medir profundidad.

## Por qué modular y no todo en main.cpp

Si ponés todo en `main.cpp` terminás con una función de 400 líneas donde:
- un bug en el guardado de archivos rompe el render
- cambiar el tamaño del tablero requiere buscar el número "9" en 15 lugares diferentes
- agregar disparidad en la siguiente fase significa reescribir todo

El principio que guía la arquitectura es **Single Responsibility**: cada clase hace exactamente
una cosa y no sabe nada de lo que no le corresponde.

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.cpp                                │
│   Sabe de: modos, threads, arranque y apagado                   │
└──────────────────┬──────────────────────────────────────────────┘
                   │ llama a
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CalibrationMode.cpp                         │
│   Sabe de: el loop de UI, teclado, ventana, orquestación        │
└──────┬─────────────────────┬────────────────────────────────────┘
       │ usa                 │ usa
       ▼                     ▼
┌──────────────┐    ┌────────────────────┐    ┌──────────────────┐
│ Chessboard   │    │ CalibrationSession │    │ CalibrationConfig│
│ Detector     │    │                    │    │                  │
│              │    │ Sabe de: archivos, │    │ Sabe de: nada.   │
│ Sabe de:     │    │ carpetas, conteo   │    │ Solo datos.      │
│ OpenCV corner│    │ de pares           │    │                  │
│ detection    │    └────────────────────┘    └──────────────────┘
└──────────────┘
```

**La regla de oro:** ninguna clase importa a otra excepto `CalibrationConfig` (que es solo datos).
`CalibrationMode` conoce a todos porque es el orquestador. Nadie más.

---

# 2. CalibrationConfig.hpp

## Qué problema resuelve

Sin este archivo, los parámetros de calibración estarían dispersos por el código:

```cpp
// Sin config — pesadilla de mantenimiento:
cv::findChessboardCorners(gray, cv::Size(9, 6), corners); // en ChessboardDetector.cpp
cv::imwrite("calib_pairs/left/...", img);                 // en CalibrationSession.cpp
int targetPairs = 30;                                     // en CalibrationMode.cpp
float squareSize = 0.025f;                                // en main.cpp
```

Si tu tablero tiene cuadrados de 30mm en lugar de 25mm, tenés que buscar `0.025` en cinco
archivos distintos, sabiendo cuál es el correcto y cuál es otra cosa.

Con `CalibrationConfig`, todos los parámetros viven en un struct, se construye una vez en
`main.cpp`, y se pasa por const-ref a todos. Cambiás un número → cambia todo el sistema.

## Anatomía del archivo

```cpp
struct CalibrationConfig {
    cv::Size boardSize{9, 6};
    float    squareSizeM = 0.025f;
    int      targetPairs = 30;
    std::string outputDir = "calib_pairs";
    cv::TermCriteria subPixCriteria{ ... };
    cv::Size subPixWinSize{11, 11};
    int      findFlags = ...;
    int      cooldownMs = 800;
};
```

### Campo por campo

---

#### `cv::Size boardSize{9, 6}`

`cv::Size` es simplemente un par de enteros: `{width, height}`.

El valor `{9, 6}` son las **esquinas internas** del tablero, NO los cuadrados. En un tablero
impreso de 10 columnas × 7 filas de cuadrados, hay 9 × 6 esquinas internas (los puntos donde
se tocan 4 cuadrados).

```
Tablero 10×7 cuadrados:

■ □ ■ □ ■ □ ■ □ ■ □
□ ■ □ ■ □ ■ □ ■ □ ■
■ □ ■ □ ■ □ ■ □ ■ □
□ ■ □ ■ □ ■ □ ■ □ ■
■ □ ■ □ ■ □ ■ □ ■ □
□ ■ □ ■ □ ■ □ ■ □ ■
■ □ ■ □ ■ □ ■ □ ■ □
         ↑
    Esquinas internas: 9 columnas × 6 filas = 54 puntos
```

> ⚠️ **Error clásico:** confundir cuadrados con esquinas. Si ponés `{10, 7}`,
> `findChessboardCorners` nunca va a encontrar el patrón y vas a pensar que
> hay un bug en tu código.

---

#### `float squareSizeM = 0.025f`

El tamaño físico de cada cuadrado en **metros**. En tu caso, 25mm = 0.025m.

Este valor no se usa en la detección de esquinas (OpenCV no sabe cuánto mide
un píxel en el mundo real). Se usa más adelante en `stereoCalibrate` para que
la traslación T salga en metros reales, no en "unidades de cuadrado".

Si lo ponés en 1.0f (la unidad arbitraria más común en tutoriales), el vector
T te dará algo como `[-2.6, 0.04, 0.03]`, que significa "2.6 cuadrados de
separación". Con 0.025f te da `[-0.065, 0.001, 0.0007]`, que significa
"-65mm de separación" — mucho más útil para `Z = f * B / d`.

---

#### `int targetPairs = 30`

Cuántos pares válidos querés recolectar antes de que el sistema declare "listo".

¿Por qué 30? La calibración de Zhang (el algoritmo de OpenCV) requiere mínimo
10-15 pares para ser estable. Con 30 tenés sobre-determinación suficiente para
que los outliers (pares mal capturados, con motion blur, etc.) no arruinen el
resultado. Más de 60 ya no mejora significativamente.

---

#### `cv::TermCriteria subPixCriteria`

Le dice a `cornerSubPix` cuándo parar de refinar. Tiene dos condiciones de parada
unidas con `+` (para cuando CUALQUIERA se cumple):

```cpp
cv::TermCriteria{
    cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
    30,      // MAX_ITER: máximo 30 iteraciones
    0.001    // EPS: parar si el movimiento fue menor a 0.001 píxeles
}
```

En la práctica, casi siempre para por EPS antes de llegar a las 30 iteraciones.

---

#### `cv::Size subPixWinSize{11, 11}`

El "vecindario" que analiza `cornerSubPix` alrededor de cada esquina candidata.
`{11, 11}` significa una ventana de 23×23 píxeles (radio 11 en cada dirección).

Más grande → más contexto → mejor para imágenes ruidosas o con baja resolución.
Más pequeño → más rápido → mejor para imágenes nítidas y de alta resolución.
Para VGA (640×480) con ESP32-CAM, `{11, 11}` es el estándar.

---

#### `int findFlags`

```cpp
int findFlags = cv::CALIB_CB_ADAPTIVE_THRESH
              | cv::CALIB_CB_NORMALIZE_IMAGE
              | cv::CALIB_CB_FAST_CHECK;
```

Son flags de bit que se combinan con OR (`|`). Cada uno activa una etapa del
pipeline interno de `findChessboardCorners`:

| Flag | Qué hace | Cuándo importa |
|---|---|---|
| `ADAPTIVE_THRESH` | Umbral local por zona, no global | Iluminación no uniforme: sombra en una esquina |
| `NORMALIZE_IMAGE` | Estira el contraste antes de umbralizar | Imagen muy oscura o muy quemada globalmente |
| `FAST_CHECK` | Rechaza rápido si no hay tablero | Ahorra ~15ms/frame cuando el tablero no está visible |

---

#### `int cooldownMs = 800`

Tiempo mínimo en milisegundos entre dos capturas consecutivas. Previene que una
sola pulsación de SPACE genere 3 o 4 pares idénticos (porque el loop corre a
~30fps y el key event puede durar varios frames).

---

### 📝 Ejercicio 2.1 — Operaciones de bit (flags)

Los flags se combinan con OR de bits. Revisá cómo funciona:

```
ADAPTIVE_THRESH = 1  = 0b00000001
NORMALIZE_IMAGE = 2  = 0b00000010
FAST_CHECK      = 8  = 0b00001000

ADAPTIVE_THRESH | NORMALIZE_IMAGE | FAST_CHECK
= 0b00000001
| 0b00000010
| 0b00001000
= 0b00001011 = 11
```

**En tu cuaderno:**
1. ¿Cuánto vale `findFlags` en decimal?
2. Si querés agregar `CALIB_CB_EXHAUSTIVE` (valor 16), ¿cómo queda la expresión?
3. ¿Cómo verificarías con código si un flag específico está activado?
   *(Pista: usá AND de bits: `if (flags & FAST_CHECK) { ... }`)*

---

# 3. ChessboardDetector

## Qué problema resuelve

Encapsula todo el pipeline de detección en una interfaz limpia:

```cpp
DetectionResult r = detector.detect(grayFrame, displayFrame);
if (r.found) { /* usar r.corners */ }
```

`CalibrationMode` no sabe si internamente se usa `findChessboardCorners` o cualquier
otro algoritmo. Si mañana OpenCV saca un `findChessboardCornersAI` más preciso,
solo cambiás el `.cpp` de esta clase.

## ChessboardDetector.hpp

```cpp
struct DetectionResult {
    bool found = false;
    std::vector<cv::Point2f> corners;
};

class ChessboardDetector {
public:
    explicit ChessboardDetector(const CalibrationConfig& config);
    DetectionResult detect(const cv::Mat& grayFrame,
                           cv::Mat&       displayFrame) const;
private:
    CalibrationConfig config_;
};
```

### `struct DetectionResult`

Un struct simple para agrupar los dos datos que produce la detección: si se encontró
el tablero, y dónde están las esquinas. Podría haberse devuelto como dos valores de
retorno con `std::pair`, pero un struct con nombres es infinitamente más legible.

`std::vector<cv::Point2f>` — cada `cv::Point2f` es un par `(x, y)` de floats que
representa una esquina en coordenadas de imagen. Hay exactamente
`boardSize.width × boardSize.height` (= 54 para 9×6) cuando `found == true`.

### `explicit ChessboardDetector(...)`

La keyword `explicit` previene conversiones implícitas. Sin ella, podrías escribir
accidentalmente `ChessboardDetector d = config;` (conversión implícita) y compilaría
sin error. Con `explicit`, solo funciona `ChessboardDetector d(config);`.

### `detect(...) const`

El `const` al final significa que el método no modifica el estado interno del objeto.
Esto es importante porque permite llamar a `detect` desde múltiples threads sobre
el mismo detector sin problemas (el objeto es efectivamente inmutable después de
construirse).

## ChessboardDetector.cpp

### Paso 1 — `findChessboardCorners`

```cpp
result.found = cv::findChessboardCorners(
    grayFrame,         // imagen de entrada (debe ser gris)
    config_.boardSize, // {9, 6} — lo que buscamos
    result.corners,    // OUTPUT: posiciones de esquinas encontradas
    config_.findFlags  // flags de optimización
);
```

💡 **¿Qué hace internamente?**

El algoritmo tiene cuatro etapas:

```
1. Umbralización adaptativa
   → convierte la imagen a blanco/negro en zonas locales
   → resiste la iluminación desigual

2. Detección de contornos
   → busca formas rectangulares del tamaño correcto

3. Quad detection
   → agrupa rectángulos en una grilla de N×M cuadriláteros

4. Verificación del patrón
   → confirma que la grilla tiene exactamente boardSize.width × boardSize.height esquinas
```

**Complejidad:** O(W × H) donde W, H son las dimensiones de la imagen. A 640×480 tarda
entre 5ms (FAST_CHECK con tablero ausente) y ~25ms (tablero presente, detección completa).

---

### Paso 2 — `cornerSubPix`

```cpp
cv::cornerSubPix(
    grayFrame,              // misma imagen gris
    result.corners,         // corners IN/OUT: entra con posición entera, sale refinada
    config_.subPixWinSize,  // {11, 11} — vecindario de análisis
    cv::Size(-1, -1),       // zero zone: sin zona muerta central
    config_.subPixCriteria  // cuándo parar
);
```

💡 **¿Por qué es necesario el refinamiento sub-pixel?**

`findChessboardCorners` localiza esquinas con precisión de ~1 píxel. Eso suena bien,
pero para calibración no lo es. Considerá:

- Una esquina detectada con error de ±1px en una imagen de 640×480
- Con focal length de 600px, eso es un error angular de `atan(1/600) ≈ 0.095°`
- A 1 metro de distancia, ese error angular produce 1.7mm de error posicional
- Multiplicado por 54 esquinas × 30 imágenes = el optimizador tiene 1620 ecuaciones
  con ruido de 1.7mm cada una — la solución de K puede tener error de varios milímetros

Con refinamiento sub-pixel el error baja a ~0.01px → 0.017mm. La diferencia en la
calidad del mapa de disparidad final es enorme.

💡 **¿Cómo funciona matemáticamente?**

En una esquina ideal, el gradiente de intensidad de la imagen (∇I) es perpendicular
a la dirección del borde. `cornerSubPix` mueve iterativamente el punto candidato
hasta que esta condición se satisface para todos los píxeles del vecindario:

```
∑ (q_i - p)ᵀ · ∇I(q_i) · ∇I(q_i)ᵀ · (q_i - p) = 0
```

Donde `p` es la posición candidata de la esquina y `q_i` son los píxeles del
vecindario. No necesitás resolver esto a mano — OpenCV lo hace — pero entender
que es un problema de **minimización iterativa** te ayuda a entender por qué
el `TermCriteria` controla cuándo parar.

---

### 📝 Ejercicio 3.1 — Precisión sub-pixel

Tu sistema tiene:
- Focal length: f = 600 px
- Baseline: B = 0.065 m
- Fórmula de profundidad: Z = f × B / d

**Sin refinamiento** (error en d de ±1 px):

A Z = 0.5m, la disparidad esperada es d = f × B / Z = 600 × 0.065 / 0.5 = 78px.

1. Calculá Z_min = f × B / (d + 1) y Z_max = f × B / (d - 1)
2. ¿Cuántos centímetros de incertidumbre en la medición de profundidad?

**Con refinamiento** (error en d de ±0.01 px):

3. Repetí el cálculo con d ± 0.01
4. ¿Cuántos milímetros de incertidumbre ahora?
5. ¿Cuánto mejora la precisión? (en porcentaje)

---

### Paso 3 — `drawChessboardCorners`

```cpp
cv::drawChessboardCorners(
    displayFrame,      // imagen COLOR donde dibuja (no la gris)
    config_.boardSize,
    result.corners,
    result.found       // true = verde y conectadas, false = rojo parcial
);
```

Dibuja los 54 puntos conectados en orden con una línea de color gradiente (rojo→verde).
Útil para verificar visualmente que la detección es correcta y que el orden de los
puntos es consistente entre la cámara izquierda y derecha (crítico para stereoCalibrate).

> ⚠️ **Por qué dibujamos sobre `displayFrame` y no sobre `grayFrame`:**
> La imagen gris es la que se pasa a los algoritmos de detección. Si dibujas sobre
> ella, los píxeles cambian y afectás la siguiente iteración del detector. Siempre
> cloná antes de dibujar.

---

# 4. CalibrationSession

## Qué problema resuelve

Todo lo relacionado con el filesystem: crear carpetas, nombrar archivos, contar pares,
guardar JPEGs. Nada más. No sabe de cámaras, de detección, ni de UI.

## CalibrationSession.hpp

```cpp
class CalibrationSession {
public:
    explicit CalibrationSession(const CalibrationConfig& config);
    bool savePair(const cv::Mat& left, const cv::Mat& right);
    int  pairCount()  const { return pairCount_; }
    bool isComplete() const { return pairCount_ >= config_.targetPairs; }
private:
    CalibrationConfig config_;
    int pairCount_ = 0;
    void        ensureDirectories() const;
    std::string leftPath(int n)    const;
    std::string rightPath(int n)   const;
};
```

### `int pairCount_ = 0`

El underscore final `_` es una convención para variables miembro privadas. Te permite
escribir `pairCount` (parámetro de función) y `pairCount_` (miembro) sin colisiones de
nombres. Es solo una convención — C++ no lo requiere — pero es ampliamente usada en
código profesional.

### Métodos inline `pairCount()` e `isComplete()`

Están definidos directamente en el `.hpp` porque son triviales (una sola expresión).
El compilador casi siempre los inlinea, eliminando el overhead de la llamada a función.

## CalibrationSession.cpp

### Constructor y `ensureDirectories()`

```cpp
CalibrationSession::CalibrationSession(const CalibrationConfig& config)
    : config_(config) {
    ensureDirectories();
}

void CalibrationSession::ensureDirectories() const {
    fs::create_directories(config_.outputDir + "/left");
    fs::create_directories(config_.outputDir + "/right");
}
```

`fs::create_directories` (de `<filesystem>`, C++17) crea la ruta completa de forma
recursiva. Si `calib_pairs/left` no existe pero tampoco `calib_pairs`, crea ambos.
Si ya existen, no hace nada (no lanza excepción). Es idempotente.

La estructura resultante en disco:

```
calib_pairs/
├── left/
│   ├── left_01.jpg
│   ├── left_02.jpg
│   └── ...
└── right/
    ├── right_01.jpg
    ├── right_02.jpg
    └── ...
```

Los sub-folders separados `left/` y `right/` son importantes: cuando ejecutes
`stereoCalibrate` vas a iterar sobre estos folders con `glob` o similar. Tener
izquierda y derecha mezclados en una sola carpeta haría ese código más complicado.

### `leftPath()` y `rightPath()`

```cpp
std::string CalibrationSession::leftPath(int n) const {
    std::ostringstream ss;
    ss << config_.outputDir << "/left/left_"
       << std::setw(2) << std::setfill('0') << n << ".jpg";
    return ss.str();
}
```

`std::setw(2)` reserva un campo de 2 caracteres. `std::setfill('0')` rellena con ceros
a la izquierda. Resultado: `left_01.jpg`, `left_09.jpg`, `left_10.jpg`.

¿Por qué el padding? Sin él, el orden alfabético del filesystem sería:
```
left_1.jpg, left_10.jpg, left_11.jpg, left_2.jpg, ...   ← INCORRECTO
```
Con padding de 2 dígitos:
```
left_01.jpg, left_02.jpg, ..., left_10.jpg, left_11.jpg  ← CORRECTO
```

Importante porque `stereoCalibrate` necesita que `left_N.jpg` corresponda exactamente
a `right_N.jpg` — el mismo par, el mismo instante.

### `savePair()`

```cpp
bool CalibrationSession::savePair(const cv::Mat& left, const cv::Mat& right) {
    int idx = pairCount_ + 1;  // próximo índice (empieza en 1, no 0)

    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 95};

    bool ok = cv::imwrite(leftPath(idx),  left,  params)
           && cv::imwrite(rightPath(idx), right, params);

    if (ok) { pairCount_++; }
    return ok;
}
```

**¿Por qué JPEG 95 y no PNG?**

- PNG es lossless pero ~5× más grande (≈3MB/par vs ≈600KB/par en JPEG 95)
- La pérdida de JPEG 95 es imperceptible para la calibración — los algoritmos
  trabajan con posiciones de esquinas, no con valores de píxel directamente
- PNG recomendado solo si tenés artefactos visibles de compresión en las esquinas

**¿Por qué `&&` y no dos `if` separados?**

`&&` cortocircuita: si `imwrite` del lado izquierdo falla, no ejecuta el del lado
derecho. Esto garantiza que nunca guardés solo una imagen del par — o ambas o ninguna.
Un par incompleto sería silenciosamente inválido y podría corromper la calibración
posterior.

---

### 📝 Ejercicio 4.1 — Zero-padding y orden lexicográfico

**En tu cuaderno:**

Tenés estos archivos sin padding: `img_1.jpg, img_2.jpg, ..., img_9.jpg, img_10.jpg`

1. Ordenalos como lo haría un sistema de archivos (orden lexicográfico, letra por letra)
2. ¿En qué posición queda `img_10.jpg`?
3. Si el algoritmo los lee en ese orden y asume que `left_N` = `right_N`, ¿qué par
   incorrecto formaría con padding faltante?
4. Con padding de 2 dígitos (`img_01.jpg` ... `img_10.jpg`), ¿el orden es correcto ahora?

---

# 5. CalibrationMode

## Qué problema resuelve

Es el orquestador: conecta streams + detector + sesión + UI. Es la única clase que
"sabe de todo" porque su trabajo específico es coordinar a los demás.

## CalibrationMode.hpp

```cpp
void runCalibrationMode(CameraStream& cam1, CameraStream& cam2,
                        const CalibrationConfig& config);
```

Una sola función libre (no es una clase). No necesita estado propio porque todo el
estado relevante vive en `CameraStream`, `CalibrationSession` y `CalibrationConfig`.
Convertirla en clase solo agregaría complejidad sin beneficio.

## CalibrationMode.cpp

### `grabFrame()` — función helper estática

```cpp
static cv::Mat grabFrame(CameraStream& stream) {
    std::lock_guard<std::mutex> lock(stream.frame_mtx);
    return stream.frame.empty() ? cv::Mat{} : stream.frame.clone();
}
```

`static` aquí significa "visible solo dentro de este .cpp". Es como un `private`
a nivel de archivo. Evita contaminar el namespace global con funciones helper que
nadie más necesita.

El `clone()` es crítico: si solo devolvieras `stream.frame`, el caller tendría una
referencia a la misma memoria que el thread de captura modifica constantemente.
`clone()` crea una copia profunda e independiente — el caller puede procesar el frame
con total seguridad, sin el mutex, todo el tiempo que quiera.

> ⚠️ **Anti-patrón frecuente:**
> ```cpp
> // INCORRECTO — el mutex se libera antes de que f sea usada:
> cv::Mat& f = stream.frame;  // referencia, no copia
> lock.unlock();
> detector.detect(f);         // race condition — otro thread puede modificar f
>
> // CORRECTO:
> cv::Mat f = stream.frame.clone();  // copia dentro del lock
> lock.unlock();                     // seguro: f es independiente
> detector.detect(f);
> ```

---

### `drawHUD()` — función helper estática

```cpp
static void drawHUD(cv::Mat& frame, bool bothValid,
                    int pairsDone, int pairsTarget, double fps)
```

Dibuja todos los elementos visuales sobre un frame. Separada como función porque
se llama dos veces por loop (una para `disp1`, otra para `disp2`) con los mismos
parámetros — elimina duplicación.

**El "pill" de status (rectángulo + texto):**

```cpp
cv::rectangle(frame,
              cv::Point(10, 8),
              cv::Point(bothValid ? 110 : 140, 48),
              labelClr, cv::FILLED);
cv::putText(frame, label, cv::Point(18, 38), ...);
```

El ancho del rectángulo cambia según si dice "VALID" (110px) o "INVALID" (140px)
para que siempre quede ajustado al texto. `cv::FILLED` como grosor significa relleno
sólido (grosor -1 también funciona y es equivalente).

---

### El loop principal de `runCalibrationMode()`

```cpp
using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::milliseconds;
auto lastCapture = Clock::now() - Ms(config.cooldownMs * 2);
```

Inicializamos `lastCapture` en el pasado (dos cooldowns atrás). Esto asegura que
la primera captura sea posible inmediatamente sin esperar 800ms al arrancar.

**El loop:**

```cpp
while (!session.isComplete()) {
    cv::Mat raw1 = grabFrame(cam1);  // copia thread-safe
    cv::Mat raw2 = grabFrame(cam2);

    if (raw1.empty() || raw2.empty()) {
        if (cv::waitKey(1) == 27) return;
        continue;  // esperar a que ambas cámaras tengan frame
    }

    // Grayscale para detección, clones color para display
    cv::Mat gray1, gray2;
    cv::cvtColor(raw1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(raw2, gray2, cv::COLOR_BGR2GRAY);

    cv::Mat disp1 = raw1.clone();
    cv::Mat disp2 = raw2.clone();

    DetectionResult r1 = detector.detect(gray1, disp1);
    DetectionResult r2 = detector.detect(gray2, disp2);
    ...
}
```

**¿Por qué dos clones?**

- `raw1`, `raw2`: imágenes originales sin tocar — estas son las que se guardan en disco
- `gray1`, `gray2`: versiones en gris para la detección
- `disp1`, `disp2`: copias color sobre las que se dibuja el HUD y las esquinas

Si dibujaras sobre `raw1` y luego lo guardaras, las esquinas azules y el texto
quedarían grabados en el JPEG de calibración. `stereoCalibrate` todavía funcionaría
(busca las esquinas, no los píxeles), pero las imágenes serían inusables para
diagnóstico visual.

---

### La captura — manejo del cooldown

```cpp
if (key == 32 && bothValid) {   // 32 = ASCII de SPACE
    auto now = Clock::now();
    if (now - lastCapture >= Ms(config.cooldownMs)) {
        lastCapture = now;
        if (session.savePair(raw1, raw2)) {
            flashCapture(combined);
        }
    }
}
```

`std::chrono::steady_clock` es un reloj monótono — siempre avanza, nunca retrocede,
no se ajusta con cambios de zona horaria ni con NTP. Ideal para medir intervalos.

La condición `now - lastCapture >= Ms(800)` compara dos `time_point` y produce una
`duration`. C++ sabe comparar duraciones de distintas unidades automáticamente.

---

### `flashCapture()` — feedback visual

```cpp
static void flashCapture(const cv::Mat& combined) {
    cv::Mat flash = combined.clone();
    cv::rectangle(flash, {0,0}, {flash.cols-1, flash.rows-1},
                  cv::Scalar(0,255,0), 10);
    cv::imshow("Stereo Calibration", flash);
    cv::waitKey(200);  // muestra el flash por 200ms
}
```

El `cv::waitKey(200)` bloquea 200ms. En ese tiempo el stream sigue recibiendo frames
(está en otro thread), pero el display queda "congelado" mostrando el flash. Cuando
el `waitKey` retorna, el loop sigue y el siguiente frame se muestra normalmente.

---

### 📝 Ejercicio 5.1 — Rastrear el ciclo de vida de un frame

Un frame tiene estas "vidas" en el sistema. En tu cuaderno, enumerá en qué
momento existe cada copia y cuándo puede ser liberada:

```
1. La ESP32 captura el frame y lo comprime como JPEG → ¿dónde vive?
2. libcurl recibe los bytes → writeCallback → ¿dónde van?
3. streamCamera encuentra SOI/EOI y llama imdecode → cv::Mat en stream.frame
4. grabFrame() clona stream.frame → raw1
5. cvtColor(raw1, gray1) → gray1
6. raw1.clone() → disp1
7. detector.detect(gray1, disp1) → corners en DetectionResult (en stack)
8. drawHUD(disp1, ...) → pixels modificados en disp1
9. hconcat(disp1, disp2, combined) → combined
10. imshow("...", combined) → pixels en buffer de ventana
11. session.savePair(raw1, raw2) → JPEG en disco
```

**Preguntas:**
1. ¿Cuántas copias del frame existen simultáneamente en el peak?
2. ¿En qué paso se puede liberar gray1?
3. ¿Por qué raw1 no puede liberarse hasta después del paso 11?
4. Si el loop corre a 30fps y cada frame es 640×480×3 bytes ≈ 900KB,
   ¿cuánta RAM usa el sistema aproximadamente en el peak?

---

# 6. main.cpp

## Estructura general

```
main()
 ├── parsea argv (--calibrate flag)
 ├── crea cam1, cam2
 ├── lanza t1, t2 (streamCamera threads — sin cambios)
 ├── warm-up loop (espera primer frame de ambas)
 ├── configura CalibrationConfig
 ├── si no --calibrate: runPreviewMode() → puede retornar si usuario presiona C
 └── si running: runCalibrationMode()
 └── shutdown: running=false, join threads
```

### El warm-up loop

```cpp
while (true) {
    bool c1ok, c2ok;
    { std::lock_guard<std::mutex> l(cam1.frame_mtx); c1ok = !cam1.frame.empty(); }
    { std::lock_guard<std::mutex> l(cam2.frame_mtx); c2ok = !cam2.frame.empty(); }
    if (c1ok && c2ok) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}
```

Sin esto, si `runCalibrationMode` arranca antes de que las cámaras tengan su primer
frame, el `if (raw1.empty() || raw2.empty()) continue` se ejecuta muchas veces y
el usuario ve una ventana en negro. Con el warm-up, la ventana aparece ya con video.

### `runPreviewMode()` y la transición de modo

```cpp
if (!startCalibration) {
    runPreviewMode(cam1, cam2);  // retorna cuando usuario presiona C o ESC
}

if (running) {
    runCalibrationMode(cam1, cam2, config);
}
```

`running` es el `std::atomic<bool>` de `StreamWorker`. Si el usuario presionó ESC
en preview, `running` se pone en false y saltamos la calibración. Si presionó C,
`running` sigue en true y entramos a calibración.

### El modo `--calibrate`

```bash
./StereoVision            # preview → C → calibración
./StereoVision --calibrate  # directo a calibración
```

Útil cuando ya verificaste que las cámaras funcionan y solo querés recolectar pares.

---

# 7. CMakeLists.txt

```cmake
set(SOURCES
    src/main.cpp
    src/StreamWorker.cpp
    src/ChessboardDetector.cpp
    src/CalibrationSession.cpp
    src/CalibrationMode.cpp
)
```

Cada `.cpp` nuevo se agrega aquí. Los `.hpp` no se agregan — CMake los encuentra
automáticamente a través de `target_include_directories`.

```cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

Esto genera `build/compile_commands.json` automáticamente cada vez que reconfigurás
cmake. Ese archivo es lo que VSCode usa para IntelliSense — nunca más tenés que
actualizar `c_cpp_properties.json` a mano.

**Para compilar:**

```bash
cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu)
```

`-j$(sysctl -n hw.logicalcpu)` detecta cuántos cores tiene tu M1 y compila en paralelo.
En un M1 Pro con 10 cores, la compilación completa tarda ~3 segundos en vez de ~20.

---

# 8. Flujo Completo de Datos

Este diagrama muestra exactamente qué dato pasa por cada función en un frame típico:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Thread streamCamera (cam1)          Thread streamCamera (cam2)          │
│  ESP32 → bytes → buffer → imdecode  ESP32 → bytes → buffer → imdecode  │
│  → stream.frame (cv::Mat BGR)        → stream.frame (cv::Mat BGR)       │
└──────────────────┬──────────────────────────────┬───────────────────────┘
                   │ grabFrame(cam1)               │ grabFrame(cam2)
                   │ [lock + clone]                │ [lock + clone]
                   ▼                               ▼
              raw1 (BGR)                      raw2 (BGR)
              640×480×3                       640×480×3
                   │                               │
          ┌────────┴──────┐              ┌─────────┴──────┐
          │ cvtColor BGR→G│              │ cvtColor BGR→G │
          ▼               ▼              ▼                ▼
       gray1            disp1         gray2            disp2
      (U8 1ch)        (BGR clone)   (U8 1ch)        (BGR clone)
          │               │              │                │
          └───────┬────────┘              └──────┬─────────┘
                  ▼                             ▼
         detector.detect()            detector.detect()
         findChessboardCorners        findChessboardCorners
         cornerSubPix                 cornerSubPix
         drawChessboardCorners→disp1  drawChessboardCorners→disp2
                  │                             │
                  ▼                             ▼
           r1.found=T/F                  r2.found=T/F
           r1.corners[54]               r2.corners[54]
                  │                             │
                  └──────────┬──────────────────┘
                             ▼
                    bothValid = r1.found && r2.found
                             │
                    drawHUD(disp1, ...)
                    drawHUD(disp2, ...)
                    hconcat(disp1, disp2) → combined
                    imshow("Stereo Calibration", combined)
                             │
                    [SPACE pressed && bothValid && cooldown ok]
                             │
                    session.savePair(raw1, raw2)
                    ├── imwrite(left_NN.jpg, raw1)   → disco
                    └── imwrite(right_NN.jpg, raw2)  → disco
```

**Nota clave:** `savePair` recibe `raw1` y `raw2` — los frames **sin** dibujo encima.
`disp1` y `disp2` solo se usan para la ventana de preview. Lo que va a disco es
siempre la imagen original.

---

# 9. Cómo Usar el Sistema

## Setup físico

Antes de capturar, hay decisiones físicas importantes:

**Separación entre cámaras (baseline):**
Tu baseline de ~70mm es razonable para objetos entre 30cm y 3m. No lo cambies
una vez que calibrés — si movés las cámaras después, la calibración queda inválida.
Fijá las cámaras físicamente (con scotch fuerte, soporte impreso en 3D, etc.).

**Orientación:**
Ambas cámaras deben apuntar en la misma dirección general, con las filas de píxeles
aproximadamente horizontales y paralelas entre sí. Una inclinación de unos pocos
grados es corregible por la calibración. Una inclinación de 45° no lo es en forma
práctica.

**Iluminación:**
Luz difusa uniforme. Evitá luz directa sobre el tablero (crea reflejos que confunden
la detección de esquinas). Una hoja de papel blanca como difusor sobre una lámpara
funciona perfectamente.

## Ejecución

```bash
# Compilar
cd build && cmake .. && make -j$(sysctl -n hw.logicalcpu)

# Correr directo en modo calibración
./StereoVision --calibrate
```

## Protocolo de captura de 30 pares

No pongas el tablero siempre al frente en el mismo lugar. El algoritmo necesita
variedad de perspectivas para calcular correctamente los parámetros de distorsión.

```
Pares  1-5:  Tablero centrado, distancias variadas (30, 40, 50, 60, 70 cm)
Pares  6-10: Inclinado horizontalmente: -30°, -15°, 0°, +15°, +30°
Pares 11-15: Inclinado verticalmente: -20°, -10°, 0°, +10°, +20°
Pares 16-20: Rotado en su propio plano (como una rueda de reloj)
Pares 21-25: Desplazado a zonas de la imagen: esquina sup-izq, sup-der, etc.
Pares 26-30: Combinaciones de inclinación + desplazamiento
```

Regla práctica: después de capturar, revisá las imágenes. Si 15 de tus 30 pares
son casi idénticos (tablero frontal centrado), el calibrado va a ser malo en los
bordes de la imagen.

## Qué hacer si la detección es inestable

| Síntoma | Causa probable | Solución |
|---|---|---|
| INVALID aunque el tablero está visible | Iluminación no uniforme | Agregar `CALIB_CB_ADAPTIVE_THRESH` (ya está en los flags) |
| Detección intermitente | Motion blur por movimiento | Sostener el tablero completamente quieto 1-2s antes de capturar |
| Solo detecta en el centro | Tablero demasiado lejos | Acercarlo hasta que ocupe al menos 1/4 del frame |
| Detección correcta pero lenta | FAST_CHECK no rechaza a tiempo | Normal — ocurre cuando el tablero está en el frame |
| Verde en una cámara, rojo en otra | Una cámara desenfocada | Revisar que ambas tengan foco en el mismo plano |

---

# 10. Qué Sigue

Una vez que tenés los 30 pares en `calib_pairs/left/` y `calib_pairs/right/`, el
siguiente módulo cargará esas imágenes y ejecutará:

```
1. calibrateCamera(puntos3D, esquinas_left,  tamaño_imagen) → K_left,  dist_left
2. calibrateCamera(puntos3D, esquinas_right, tamaño_imagen) → K_right, dist_right
3. stereoCalibrate(...)  → R, T, E, F
4. stereoRectify(...)    → R1, R2, P1, P2, Q
5. initUndistortRectifyMap(...)  → mapas de reproyección
6. Guardar todo en stereo_calib.yaml
```

El archivo YAML resultante es el que cargará el módulo de disparidad (SGBM) para
rectificar cada frame en tiempo real y calcular la profundidad Z.

## Checklist antes de pasar al siguiente módulo

- [ ] ¿Tenés exactamente N pares en `left/` y N pares en `right/` con los mismos índices?
- [ ] ¿Las imágenes tienen variedad de ángulos (no todas frontales)?
- [ ] ¿El tablero ocupa al menos 1/4 del frame en la mayoría de los pares?
- [ ] ¿Las imágenes están nítidas (sin motion blur visible)?
- [ ] ¿El tablero aparece completo (sin esquinas cortadas por el borde de la imagen)?

Si marcás todo, estás listo para la fase de calibración y rectificación.

---

*Proyecto: Visión Artificial — Universidad Politécnica Salesiana*
*Período Lectivo: Abril – Agosto 2026*
*Docente: Ing. Vladimir Robles Bykbaev*
