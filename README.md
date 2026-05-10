# 📷 Proyecto Integrador — Visión Estéreo con ESP32-XIAO-S3
### Guía de aprendizaje progresiva: Etapa 0 (Streaming) → Etapa 1 (Calibración)

> **Cómo usar este documento**
> Este README no es solo documentación. Es una guía de estudio.
> Cada sección tiene: Marco Teórico → Marco Técnico → Ejemplo concreto → Ejercicio de papel y lápiz.
> Los ejercicios con 📝 son para tu cuaderno. No los saltes — son lo que hace que el conocimiento sea tuyo.

---

# MÓDULO 0 — Lo que ya construimos: El sistema de streaming

Antes de calibrar nada, necesitás entender exactamente qué hace el código que ya funciona. No es solo "mostrar cámaras". Hay conceptos profundos ahí adentro.

---

## 0.1 — ¿Qué es un stream MJPEG?

### Marco Teórico

La ESP32-CAM no transmite video como tal. Transmite algo llamado **Motion JPEG (MJPEG)**: una secuencia continua de imágenes JPEG, una tras otra, dentro de una misma conexión HTTP.

Pensá en un libro de caricaturas: cada página es una imagen estática, y cuando las pasas rápido ves "movimiento". Eso es MJPEG. No es video comprimido como H.264 — es literalmente JPEGs concatenados.

### Marco Técnico: Anatomía de un archivo JPEG

Todo archivo JPEG en el mundo, sin excepción, empieza y termina con dos bytes específicos:

```
Inicio (SOI - Start of Image):  FF D8
Final  (EOI - End of Image):    FF D9
```

Estos son los "marcadores" del estándar JPEG (ISO 10918). Son el equivalente a las comillas en texto: `"esto es un JPEG"`.

Un stream MJPEG se ve así en memoria, byte por byte:

```
... FF D8 [datos del frame 1] FF D9  FF D8 [datos del frame 2] FF D9  FF D8 ...
         |←————— JPEG 1 ——————→|     |←————— JPEG 2 ——————→|
```

El problema es que los datos llegan en **fragmentos** (chunks) de red, no perfectamente alineados. Puede llegar la mitad de un JPEG en un chunk y la otra mitad en el siguiente. Tu código tiene que:
1. Acumular datos en un buffer
2. Buscar el SOI y EOI
3. Extraer exactamente esos bytes
4. Decodificar el JPEG
5. Limpiar el buffer

### El código hace esto aquí:
```cpp
// Busca FF D8 en el buffer acumulado
auto soi = std::search(localBuffer.begin(), localBuffer.end(),
                       JPEG_SOI, JPEG_SOI + 2);

// Busca FF D9 *después* del SOI (fix del Bug 4)
auto eoi = std::search(soi + 2, localBuffer.end(),
                       JPEG_EOI, JPEG_EOI + 2);

// Extrae exactamente esos bytes y decodifica
std::vector<uchar> jpg(soi, eoi + 2);
cv::Mat img = cv::imdecode(jpg, cv::IMREAD_COLOR);
```

---

### 📝 Ejercicio 0.1 — Sistema numérico hexadecimal

El marcador SOI es `0xFF 0xD8`. Para entenderlo tenés que saber hexadecimal.

**En tu cuaderno, convertí estos números hex a decimal:**

| Hexadecimal | Decimal | Razonamiento |
|---|---|---|
| `0xFF` | ? | F = 15, entonces: 15×16 + 15×1 = ? |
| `0xD8` | ? | D = 13, entonces: 13×16 + 8×1 = ? |
| `0xD9` | ? | D = 13, entonces: 13×16 + 9×1 = ? |

*Respuestas: 255, 216, 217. ¿Lo notás? 216 y 217 son consecutivos, por eso SOI y EOI son tan fáciles de recordar.*

**Ejercicio inverso — convertí de decimal a hex:**

- 100 decimal = `0x??`  *(Dividí 100 entre 16 → cociente 6, resto 4 → 0x64)*
- 255 decimal = `0x??`
- 200 decimal = `0x??`

---

### 📝 Ejercicio 0.2 — Búsqueda lineal (lo que hace `std::search`)

`std::search` hace una búsqueda de subsecuencia. Implementala a mano para entenderla.

**Dado este buffer:**
```
buffer = [10, 20, 0xFF, 0xD8, 45, 67, 88, 0xFF, 0xD9, 12]
patron_SOI = [0xFF, 0xD8]
```

En tu cuaderno, simulá el algoritmo posición por posición:
- Posición 0: ¿buffer[0..1] == [0xFF, 0xD8]? → 10, 20 → No
- Posición 1: ¿buffer[1..2] == [0xFF, 0xD8]? → 20, 0xFF → No
- Posición 2: ¿buffer[2..3] == [0xFF, 0xD8]? → ¿?

**Preguntas:**
1. ¿En qué posición encontraste el SOI?
2. ¿En qué posición encontraste el EOI?
3. ¿Cuántos bytes tiene el JPEG (SOI hasta EOI inclusive)?

---

## 0.2 — Hilos (Threads) y por qué los necesitamos

### Marco Teórico

Un hilo es una "tarea paralela". Tu CPU puede hacer varias cosas "al mismo tiempo" (en realidad las alterna muy rápido, pero el efecto es el mismo).

**¿Por qué necesitamos dos hilos para cada cámara?**

Problema: `curl_easy_perform()` es una función **bloqueante**. Significa que mientras está descargando datos de la cámara, el programa se "congela" ahí y no puede hacer nada más.

Si pusieras todo en un solo hilo:
```
[Descargar cam1] → congela → [Descargar cam2] → congela → [Mostrar] 
```
Nunca podrías mostrar ambas cámaras simultáneamente.

Con hilos:
```
Hilo principal:  [Mostrar frames] → [Mostrar frames] → [Mostrar frames]
Hilo cam1:       [Descargar] → [Descargar] → [Descargar] → ...
Hilo cam2:       [Descargar] → [Descargar] → [Descargar] → ...
```

### Marco Técnico: El problema de los datos compartidos (Race Condition)

Aquí viene el peligro. Imaginá dos personas (hilos) editando el mismo documento de Word al mismo tiempo sin coordinarse. El resultado sería basura.

Una **race condition** es cuando dos hilos acceden a los mismos datos sin protección, y el resultado depende de quién llegue primero — lo cual es impredecible.

En nuestro código, el buffer es compartido entre el hilo de curl (que escribe) y el hilo de procesamiento (que lee). Sin protección:

```
Hilo curl:      Escribiendo byte 500 del buffer...
Hilo proceso:   Leyendo buffer... (¡está a medias!)
                → Decodifica JPEG corrupto → Imagen basura o crash
```

La solución es el **mutex** (Mutual Exclusion — exclusión mutua):

```cpp
std::mutex buffer_mtx;  // Es como un semáforo físico

// Solo UN hilo a la vez puede estar dentro de este bloque:
{
    std::lock_guard<std::mutex> lock(stream->buffer_mtx);  // Cierra el semáforo
    stream->buffer.insert(...);  // Escribe seguro
}  // Aquí se destruye lock → abre el semáforo automáticamente
```

`std::lock_guard` usa RAII (Resource Acquisition Is Initialization): el mutex se libera automáticamente cuando el objeto sale de scope. No podés olvidarte de liberarlo.

---

### 📝 Ejercicio 0.3 — Race Condition en papel

Simulá este escenario con dos "personas" (A = curl, B = proceso):

El buffer tiene capacidad para 10 elementos. Empieza vacío: `[]`

**Instrucciones de A (curl):** Insertar [1, 2, 3, 4, 5] una por una.
**Instrucciones de B (proceso):** Cuando haya al menos 3 elementos, leer el buffer completo.

Escenario 1 — Sin mutex (pueden interrumpirse entre cualquier instrucción):
```
A inserta 1 → buffer = [1]
A inserta 2 → buffer = [1, 2]
B lee buffer = [1, 2]  ← ¿Tiene el JPEG completo? No → imagen corrupta
A inserta 3 → buffer = [1, 2, 3]
A inserta 4 → buffer = [1, 2, 3, 4]
B lee buffer = [1, 2, 3, 4]  ← ¿Cuál lectura es "la buena"?
```

Escenario 2 — Con mutex:
```
A toma mutex → inserta 1, 2, 3, 4, 5 → libera mutex
B espera...                           B toma mutex → lee [1,2,3,4,5] → libera mutex
```

**Pregunta:** ¿En qué escenario el resultado es siempre predecible? ¿Por qué?

---

### 📝 Ejercicio 0.4 — El Bug 3 en papel (el más importante)

Este bug podría causar un crash difícil de reproducir (el peor tipo).

**Escenario del código original:**
```
streamCamera() crea curlThread
curlThread: ejecuta curl_easy_perform() — tarda 5 segundos en correr
streamCamera(): llega a curl_easy_cleanup() — lo ejecuta INMEDIATAMENTE
              → curl_easy_cleanup limpia la memoria que curlThread está usando
curlThread: intenta acceder a esa memoria → CRASH (o comportamiento indefinido)
```

**En tu cuaderno, dibujá una línea de tiempo:**
- Eje X = tiempo (0s a 6s)
- Línea superior = curlThread
- Línea inferior = streamCamera()
- Marcá dónde ocurre el crash en el código original
- Marcá por qué el código corregido lo evita

*Pista: en el código corregido, `curl_easy_perform()` corre DIRECTAMENTE en el hilo de `streamCamera()`, entonces `cleanup()` no puede ejecutarse antes de que `perform()` termine.*

---

## 0.3 — La señal de parada: `std::atomic<bool>`

### Marco Teórico

Cuando el usuario presiona ESC, el main pone `running = false`. Los hilos de cámara tienen que detectar eso y terminar.

**¿Por qué `atomic` y no simplemente `bool`?**

Una variable `bool` en C++ no es atómica por defecto. Significa que modificarla o leerla puede no ser "instantáneo" — puede tomar varios pasos del procesador. Si un hilo está en medio de escribir `false` y otro lee a la mitad, el resultado es indefinido.

`std::atomic<bool>` garantiza que la lectura y escritura son una operación indivisible. No hay "a medias".

---

### 📝 Ejercicio 0.5 — Diseño de señales entre hilos

Sin escribir código, describí en texto o diagrama cómo comunicarías "parar" en estos escenarios:

1. **Escenario A:** Tenés 4 hilos que llenan una cola. Quiero que todos paren cuando la cola llegue a 1000 elementos. ¿Qué variable compartida usarías y quién la modifica?

2. **Escenario B:** Tenés un hilo que descarga imágenes y otro que las procesa. El hilo de proceso detecta que la imagen es inválida 10 veces seguidas y quiere decirle al de descarga que cambie de URL. ¿Cómo diseñarías esa comunicación?

3. **Escenario C (el nuestro):** ¿Por qué retornar `0` en `writeCallback` es suficiente para parar `curl_easy_perform`? ¿Qué tiene que hacer libcurl internamente cuando el callback dice "no quiero más datos"?

---

---

# MÓDULO 1 — Calibración Estéreo

> **Contexto de por qué llegamos aquí:**
> Ahora tenés video de ambas cámaras. Pero si medís la distancia ahora, va a estar completamente mal. ¿Por qué? Porque las cámaras no son perfectas, no están perfectamente alineadas, y el software no sabe qué "forma" tiene la lente ni cómo están posicionadas una respecto a la otra. La calibración resuelve todo eso.

---

## 1.1 — El modelo Pinhole (el fundamento de todo)

### Marco Teórico

Antes de hablar de calibración, tenés que entender cómo una cámara forma una imagen matemáticamente.

El **modelo Pinhole** (cámara estenopeica) es la simplificación matemática fundamental. Imaginate una caja cerrada con un agujero infinitamente pequeño en el frente. La luz que entra proyecta la escena en la pared del fondo, invertida.

La pregunta central es: **dado un punto 3D en el mundo real, ¿dónde cae en la imagen 2D?**

### La proyección en coordenadas

Un punto 3D: `P = (X, Y, Z)` donde Z es la profundidad (distancia de la cámara).

Su proyección en el plano de imagen (en píxeles): `p = (u, v)`

La relación es:
```
u = f * (X / Z)
v = f * (Y / Z)
```

Donde `f` es la **distancia focal** — qué tan "lejos" está el plano de imagen del agujero.

**Intuición:** Dividís por Z porque objetos más lejos (Z grande) se ven más pequeños. Si algo está el doble de lejos, ocupa la mitad de píxeles. Esto es la perspectiva.

---

### 📝 Ejercicio 1.1 — Proyección perspectiva básica

Tenés una cámara con distancia focal `f = 500 píxeles`.

Un objeto está en `P1 = (2, 3, 10)` (X=2m, Y=3m, Z=10m de profundidad).

**Calculá:**
1. ¿En qué coordenada (u, v) aparece en la imagen?
2. El mismo objeto se mueve al doble de distancia: `P2 = (2, 3, 20)`. ¿Nueva coordenada?
3. ¿Cuánto más pequeño (en píxeles) se ve en P2 vs P1?
4. Si el objeto mide 1m × 1m en el mundo real, ¿cuántos píxeles ocupa en cada caso?

*Pista: tamaño en píxeles = tamaño_real × f / Z*

---

### Marco Técnico: La Matriz Intrínseca K

En la realidad, el modelo es un poco más complejo porque:
- La distancia focal puede ser diferente en X e Y (por píxeles no cuadrados): `fx`, `fy`
- El centro óptico no es exactamente el centro de la imagen: `cx`, `cy`
- Hay que pasar a coordenadas de píxeles, no solo metros

Esto se representa con la **matriz intrínseca K** (también llamada matriz de cámara):

```
    | fx   0   cx |
K = |  0  fy   cy |
    |  0   0    1 |
```

La proyección completa usando álgebra lineal (coordenadas homogéneas):

```
    |u|       |X|
s * |v| = K * |Y|
    |1|       |Z|
```

Donde `s` es un escalar (la profundidad Z). Desarrollando:

```
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

**¿Qué significa cada parámetro?**

- **fx, fy** — Distancia focal en píxeles. Para una ESP32-CAM típica, algo entre 400-800 píxeles. Es "cuántos píxeles" equivale a 1 metro a 1 metro de distancia.
- **cx, cy** — Centro óptico principal. En teoría es el centro de la imagen (ej: 320, 240 para una imagen 640×480), pero en la práctica puede estar desplazado unos píxeles.

**K es única para cada cámara física.** Si la cámara cae al suelo o cambiás el zoom, K cambia. La calibración determina K.

---

### 📝 Ejercicio 1.2 — Trabajar con la matriz K

Tenés esta matriz de cámara (valores realistas para ESP32-CAM a 640×480):
```
    | 600    0   320 |
K = |   0  600   240 |
    |   0    0     1 |
```

**Parte A:** Calculá la proyección del punto `P = (0.5, -0.3, 2.0)` metros.

Usá:
```
u = fx * (X/Z) + cx = 600 * (0.5/2.0) + 320 = ?
v = fy * (Y/Z) + cy = 600 * (-0.3/2.0) + 240 = ?
```

**Parte B:** ¿El punto está dentro de la imagen (0 ≤ u ≤ 640, 0 ≤ v ≤ 480)?

**Parte C:** Ahora el punto se mueve a `P2 = (0.5, -0.3, 4.0)`. ¿Nueva proyección? ¿Qué observás?

**Parte D (al revés — reprojeción):** Te dan el punto en imagen `(u, v) = (470, 150)` y sabés que el objeto está a `Z = 3m`. ¿Cuáles son sus coordenadas 3D (X, Y)?

*Pista para D:* 
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```
*Nota: esto es exactamente lo que hace OpenCV cuando calculás profundidad desde disparidad — esta fórmula es la esencia del proyecto.*

---

## 1.2 — Distorsión de la lente

### Marco Teórico

El modelo Pinhole asume que la lente es perfecta. No lo es. Las lentes físicas doblan la luz de manera no uniforme, especialmente en los bordes. Esto se llama **distorsión**.

Hay dos tipos principales:

**1. Distorsión radial** — La imagen se "abomba" o se "hunde" desde el centro.
- Distorsión de barril (barrel): líneas rectas se curvan hacia afuera (típica en lentes gran angular como la ESP32-CAM)
- Distorsión de cojín (pincushion): líneas rectas se curvan hacia adentro (típica en teleobjetivos)

**2. Distorsión tangencial** — La lente no está perfectamente paralela al sensor. Produce un efecto de "inclinación".

La corrección matemática usa estos coeficientes: `(k1, k2, p1, p2, k3)`

Donde k1, k2, k3 son coeficientes radiales y p1, p2 son tangenciales.

La corrección se aplica así (simplificado):
```
r² = x² + y²  (distancia al centro, normalizada)

x_corregido = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
y_corregido = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
```

**Intuición:** La fórmula toma cada píxel distorsionado y lo "mueve" a su posición correcta. Esto es lo que hace `cv::undistort()`.

---

### 📝 Ejercicio 1.3 — Intuición sobre distorsión radial

Tomá una hoja cuadriculada. Dibujá un círculo en el centro.

1. Ahora "deformá" el dibujo como si fuera distorsión de barril: las líneas rectas de la cuadrícula se curvan alejándose del centro. Describí con palabras qué pasa con una línea horizontal que cruza el borde de la imagen.

2. Con k1 = 0.1 (distorsión suave) y un punto en `x = 0.8, y = 0`:
   - Calculá `r² = x² + y² = ?`
   - Calculá el factor de corrección: `1 + k1*r² = ?`
   - ¿El punto se mueve hacia adentro o hacia afuera del centro?

3. ¿Por qué la distorsión en el centro de la imagen es casi cero pero aumenta hacia los bordes? *(Pista: mirá la fórmula, ¿qué pasa con r cuando estás en el centro?)*

---

## 1.3 — La geometría estéreo: ¿cómo dos cámaras dan profundidad?

### Marco Teórico: El principio de la disparidad

Los humanos tenemos dos ojos separados ~65mm. Cada ojo ve la escena desde un ángulo ligeramente diferente. El cerebro calcula la profundidad comparando las diferencias entre lo que ve cada ojo.

La visión estéreo artificial hace exactamente lo mismo.

**El concepto clave: disparidad**

Si un objeto está a distancia `Z`, aparece en:
- Cámara izquierda: posición `uL`
- Cámara derecha: posición `uR`

La **disparidad** es: `d = uL - uR`

La relación con la profundidad es:
```
Z = f * B / d
```

Donde:
- `Z` = profundidad en metros
- `f` = distancia focal en píxeles
- `B` = baseline = distancia entre las dos cámaras en metros
- `d` = disparidad en píxeles

**Intuición crucial:**
- Objeto cercano → disparidad grande (gran diferencia entre posiciones)
- Objeto lejano → disparidad pequeña (casi el mismo lugar en ambas cámaras)
- Objeto infinitamente lejos → disparidad = 0

---

### 📝 Ejercicio 1.4 — La fórmula de profundidad

Tenés tu sistema estéreo con:
- `f = 600 píxeles`
- `B = 0.065 m` (65mm de separación entre cámaras — típico para ESP32)

**Calculá la profundidad para estas disparidades:**

| Disparidad (píxeles) | Z = f × B / d |
|---|---|
| d = 60 | Z = 600 × 0.065 / 60 = **0.65m** (65cm) |
| d = 30 | Z = ? |
| d = 15 | Z = ? |
| d = 6  | Z = ? |
| d = 2  | Z = ? |

**Observá el patrón:** ¿Qué pasa con la precisión cuando el objeto está muy lejos? *(Pista: si Z cambia de 3.25m a 6.5m, ¿cuánto cambia d? ¿Es fácil medir ese cambio con precisión?)*

**Ejercicio inverso:** Querés medir la profundidad con error máximo de ±1cm. El objeto está a 50cm.
- ¿Cuál es la disparidad esperada a 50cm?
- Si la disparidad tiene error de ±0.5 píxeles, ¿cuánto error en centímetros da eso?

---

### Marco Técnico: La Matriz Esencial E y la Matriz Fundamental F

Cuando tenés dos cámaras, hay una restricción geométrica muy útil: si sabés dónde está un punto en la imagen izquierda, **no podés encontrarlo en cualquier lugar** de la imagen derecha — solo en una línea específica llamada **línea epipolar**.

**La Matriz Esencial E** captura la relación geométrica entre dos cámaras en coordenadas normalizadas (sin los parámetros intrínsecos):

```
p_derecha^T * E * p_izquierda = 0
```

**La Matriz Fundamental F** es similar pero trabaja directamente en coordenadas de píxeles:
```
F = K_derecha^(-T) * E * K_izquierda^(-1)
```

**¿Por qué importan?** La calibración estéreo determina E y F, lo que permite rectificar las imágenes: hacer que las líneas epipolares sean perfectamente horizontales. Después de la rectificación, para encontrar correspondencias entre las cámaras **solo tenés que buscar en la misma fila**, no en toda la imagen. Esto acelera enormemente el cálculo de disparidad.

---

### La relación R, T entre cámaras

La calibración estéreo también determina:
- **R** (Rotation matrix 3×3): cómo está rotada la cámara derecha respecto a la izquierda
- **T** (Translation vector 3×1): cuánto está desplazada la cámara derecha respecto a la izquierda

Idealmente:
```
R ≈ Identidad (las cámaras apuntan exactamente en la misma dirección)
    | 1 0 0 |
R = | 0 1 0 |
    | 0 0 1 |

T ≈ (-B, 0, 0)  (solo desplazamiento horizontal, el "baseline")
    | -0.065 |
T = |    0   |
    |    0   |
```

En la realidad, R no es exactamente identidad porque montar dos cámaras perfectamente paralelas es imposible a mano. La calibración mide exactamente cuánto se desvían.

---

### 📝 Ejercicio 1.5 — Matrices de rotación básicas

Una matriz de rotación R describe una rotación 3D. La rotación en Z (girar la cámara como si la rotaras en el plano horizontal) es:

```
        | cos(θ)  -sin(θ)  0 |
Rz(θ) = | sin(θ)   cos(θ)  0 |
        |    0       0     1 |
```

**Tu cámara derecha está girada 2° respecto a la izquierda. θ = 2° = 0.035 radianes.**

1. Calculá `cos(0.035) ≈ ?` y `sin(0.035) ≈ ?` *(Pista: para ángulos pequeños, cos(θ) ≈ 1 y sin(θ) ≈ θ)*

2. ¿Cómo se ve la matriz Rz(2°)?

3. Un punto en la cámara izquierda está en `P = (0.3, 0.0, 2.0)`. Aplicá la rotación: `P' = R * P`

4. ¿Cuántos píxeles se desplaza en la imagen? (usá f = 600px)

*Esto ilustra por qué la calibración importa: 2° de desviación en la cámara puede causar varios píxeles de error en la imagen, lo que se traduce en error en la profundidad.*

---

## 1.4 — El tablero de ajedrez: ¿por qué funciona?

### Marco Teórico

El tablero de ajedrez es el "calibrador universal" de visión por computador. ¿Por qué?

1. **Detección sub-pixel automática:** Las esquinas del tablero (donde el negro toca el blanco) forman un patrón de alto contraste que OpenCV puede localizar con precisión sub-pixel usando `cv::findChessboardCorners()` + `cv::cornerSubPix()`.

2. **Coordenadas 3D conocidas:** Si el cuadrado mide `s` metros, entonces las esquinas están en posiciones exactamente conocidas: `(0,0,0), (s,0,0), (2s,0,0), ..., (0,s,0), (s,s,0), ...`

3. **Muchas ecuaciones:** Cada esquina detectada da 2 ecuaciones (u y v). Un tablero 9×6 tiene 54 esquinas internas = 108 ecuaciones. La matriz K tiene solo 5 parámetros desconocidos. Con muchas imágenes (mínimo 20), el sistema está muy sobre-determinado y la solución es robusta.

4. **Variación de perspectiva:** Al mover el tablero en diferentes posiciones y ángulos, estás dándole al algoritmo información desde múltiples puntos de vista, lo que hace la solución mucho más robusta.

### El proceso matemático (simplificado)

El algoritmo de calibración de Zhang (el que usa OpenCV) funciona así:

```
Para cada imagen del tablero:
  1. Detectar esquinas en imagen (u_i, v_i)
  2. Conocer posición 3D real (X_i, Y_i, 0)
  3. La relación es: [u_i, v_i] = f(K, dist, R_img, t_img, X_i, Y_i, 0)

Minimizar el error de reproyección total:
  E = Σ ||[u_i, v_i] - proyección(K, dist, R_img, t_img, [X_i, Y_i, 0])||²

Resolver con optimización no-lineal (Levenberg-Marquardt)
→ K, dist, R, T
```

**Error de reproyección:** Después de calcular K, re-proyectás los puntos 3D conocidos a la imagen y medís cuántos píxeles se alejan de las esquinas detectadas. Un buen calibrado da error < 0.5 píxeles.

---

### 📝 Ejercicio 1.6 — El error de reproyección

Tenés un tablero con cuadrados de 25mm. Una esquina está en posición 3D real `(0.025, 0.050, 0)` metros.

Con una K estimada:
```
    | 590    0   315 |
K = |   0  590   245 |
    |   0    0     1 |
```

Y el tablero a `Z = 0.5m` (simplificando, sin rotación):

1. Calculá la posición proyectada `(u, v)`.
2. En la imagen detectaste la esquina en `(u_det, v_det) = (312, 306)`.
3. Calculá el error de reproyección: `e = sqrt((u - u_det)² + (v - v_det)²)` píxeles.
4. ¿Este error es aceptable (< 0.5 píxeles) o indica que K necesita ajuste?

---

## 1.5 — Tu pregunta sobre la iluminación: respuesta real

> *"aveces de dia aveces de noche... y el dia de la presentacion lo hare en la uni... esto variara mucho..."*

Tu intuición de "calibrar en cada entorno" está bien orientada, pero hay matices importantes.

### ¿Por qué la iluminación afecta la calibración?

La calibración NO usa colores ni intensidades — usa la **posición geométrica** de las esquinas. Pero la iluminación sí afecta la **detección de esquinas**:

- Con poca luz → imagen ruidosa → esquinas detectadas con error de varios píxeles → K menos precisa
- Con luz muy directa → reflejos en el papel → algunas esquinas no se detectan
- Con iluminación no uniforme → contraste variable entre cuadrados → detección inestable

### Estrategia práctica recomendada

**Opción A — Un solo conjunto de parámetros (la mejor para el proyecto):**
Calibrar en condiciones "intermedias-buenas" con buena iluminación, en un ambiente parecido al de la presentación. Las matrices K e intrínsecas son **propiedades físicas de la lente**, no cambian con la iluminación. Lo que sí cambia es la calidad de detección al recalibrar.

**En la práctica:** Si tu tablero es detectable (contraste suficiente), los resultados K, dist, R, T serán muy similares independientemente de la iluminación — con diferencias en el tercer decimal. La calibración es robusta.

**Opción B — Lo que realmente necesitás ajustar por entorno:**
No la calibración, sino el preprocesamiento. Según la guía, aplicarás **CLAHE** (Contrast Limited Adaptive Histogram Equalization) ANTES de calcular la disparidad. CLAHE se adapta automáticamente a la iluminación local. Eso compensa mucho las variaciones.

**Recomendación concreta:**
1. Calibrá una vez con buena iluminación → guardá en YAML
2. En presentación: usá el mismo YAML
3. Ajustá los parámetros de CLAHE si la imagen se ve muy oscura o muy quemada
4. Si hay reflejos en el tablero: imprimir en papel mate, no brillante

**La clave es esto:** CLAHE es tu seguro de iluminación para el algoritmo de disparidad. La calibración intrínseca es estable.

---

### 📝 Ejercicio 1.7 — Por qué CLAHE es mejor que ecualización global

La ecualización de histograma global redistribuye los píxeles para que todos los niveles de brillo estén igual de representados. El problema es que una zona oscura pequeña puede dominar el histograma global y hacer que el resto de la imagen quede mal.

**Simulación en papel:**

Imagen de 8 píxeles con valores: `[200, 210, 205, 208, 5, 3, 7, 4]`
- Los primeros 4 son la escena bien iluminada
- Los últimos 4 son una sombra muy oscura

1. Calculá el histograma (qué valores aparecen y cuántas veces)
2. Si ecualizás globalmente, los valores del 200-210 quedan "comprimidos" porque ocupan solo 4 de 8 píxeles. ¿Qué le pasa al contraste en esa zona?
3. CLAHE trabaja por zonas (tiles): ecualiza los primeros 4 píxeles independientemente de los últimos 4. ¿Por qué eso preserva mejor el contraste local?

---

## 1.6 — Protocolo de captura para calibración

### Lo que vas a hacer (código viene después, esto es el proceso lógico):

```
PARA cada posición (mínimo 25, recomendado 40):
  1. Sostener tablero frente a ambas cámaras simultáneamente
  2. Esperar frame estable (sin motion blur)
  3. Capturar par sincronizado (img_izq, img_der) con el MISMO timestamp
  4. Verificar que OpenCV detecta las esquinas en AMBAS imágenes
  5. Si y solo si detecta en ambas → guardar el par
  6. Ir a la siguiente posición
```

### Posiciones recomendadas para cobertura óptima:

Necesitás cubrir el volumen de la imagen, no solo el centro:

```
Posición  1-5:  Tablero centrado, diferente distancia (30cm, 50cm, 70cm, 90cm, 110cm)
Posición  6-10: Tablero inclinado IZQUIERDA-DERECHA (-30°, -15°, 0°, +15°, +30°)
Posición 11-15: Tablero inclinado ARRIBA-ABAJO (-20°, -10°, 0°, +10°, +20°)
Posición 16-20: Tablero rotado en su plano (como si lo giraras en el reloj)
Posición 21-25: Tablero en esquinas de la imagen (arriba-izq, arriba-der, abajo-izq, abajo-der, centro)
Posición 26-30: Combinaciones de lo anterior
```

**Regla de oro:** Si todas tus capturas son con el tablero perfectamente frontal y centrado, tu K será precisa para ese ángulo pero mala para los bordes de la imagen. ¡Cubrir los bordes es crítico!

---

### 📝 Ejercicio 1.8 — Por qué necesitamos múltiples poses

Tenés 5 parámetros desconocidos en K: `fx, fy, cx, cy` y al menos `k1, k2` de distorsión = 6 mínimo.

Cada par de esquinas detectadas (u, v) te da 2 ecuaciones.

1. Con 1 imagen del tablero 9×6 (54 esquinas internas): ¿cuántas ecuaciones tenés?
2. ¿Está el sistema determinado o sobre-determinado con solo 1 imagen?
3. ¿Por qué queremos sobre-determinación en lugar de exactamente suficientes ecuaciones? *(Pista: ruido de medición)*
4. Con 30 imágenes del tablero, ¿cuántas ecuaciones en total? ¿Qué tan sobre-determinado está el sistema?

---

## 1.7 — Los outputs de la calibración y cómo usarlos

Después de ejecutar `cv::calibrateCamera()` y `cv::stereoCalibrate()`, obtenés:

### Para cada cámara individualmente:
```cpp
cv::Mat K_left, dist_left;   // Matriz intrínseca + distorsión cámara izquierda
cv::Mat K_right, dist_right; // Ídem para derecha
```

### Del sistema estéreo:
```cpp
cv::Mat R;  // Rotación entre cámaras (3×3)
cv::Mat T;  // Traslación entre cámaras (3×1) — el baseline está aquí
cv::Mat E;  // Matriz esencial (3×3)
cv::Mat F;  // Matriz fundamental (3×3)
```

### De la rectificación (`cv::stereoRectify`):
```cpp
cv::Mat R1, R2;   // Rotación para rectificar cada cámara
cv::Mat P1, P2;   // Matrices de proyección después de rectificar
cv::Mat Q;        // Matriz de reproyección 3D (¡la más importante para calcular Z!)
```

**La matriz Q es la que usarás para convertir disparidad a profundidad real.** Tiene la forma:
```
    | 1  0   0   -cx        |
Q = | 0  1   0   -cy        |
    | 0  0   0    f         |
    | 0  0  -1/B  (cx-cx')/B|
```

Donde B es el baseline. Cuando ejecutás `cv::reprojectImageTo3D(disparidad, puntos3D, Q)`, internamente hace:

```
Z = f * B / d  ← ¡la fórmula del Ejercicio 1.4!
```

---

### 📝 Ejercicio 1.9 — Rastrear la lógica completa

En tu cuaderno, dibujá este diagrama y completá cada flecha con la operación matemática:

```
Tablero físico         →  [?]  →  Esquinas en imagen (u,v)
Posiciones 3D conocidas→  [?]  →  Ecuaciones de proyección
Muchas ecuaciones      →  [?]  →  K, dist (calibración)
K_left + K_right       →  [?]  →  R, T, E, F (geometría estéreo)
R, T, K                →  [?]  →  R1, R2, P1, P2, Q (rectificación)
Frame izq + Frame der  →  [?]  →  Mapa de disparidad
Disparidad + Q         →  [?]  →  Profundidad Z en metros
```

Cada [?] es el nombre de la operación o función de OpenCV. Podés buscar: `findChessboardCorners`, `calibrateCamera`, `stereoCalibrate`, `stereoRectify`, `StereoSGBM`, `reprojectImageTo3D`.

---

## 1.8 — El archivo de calibración YAML

Todos estos datos se guardan en un archivo YAML (requerimiento de la guía del proyecto). Vas a ver algo así:

```yaml
camera_matrix_left: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [ 598.3, 0, 317.2, 0, 599.1, 241.8, 0, 0, 1 ]

distortion_coeffs_left: !!opencv-matrix
  rows: 1
  cols: 5
  dt: d
  data: [ -0.42, 0.19, 0.001, -0.002, -0.05 ]

R: !!opencv-matrix
  rows: 3
  cols: 3
  ...

T: !!opencv-matrix
  rows: 3
  cols: 1
  data: [ -0.0632, 0.0012, 0.0008 ]  # ≈ -63.2mm de baseline, casi puro en X
```

Leé el T: el primer valor es -63.2mm — eso es tu baseline. Los otros dos valores (1.2mm, 0.8mm) son las pequeñas imperfecciones de montaje.

---

### 📝 Ejercicio 1.10 — Verificar que la calibración tiene sentido

Dado este vector T del ejercicio anterior:
```
T = [-0.0632, 0.0012, 0.0008]
```

1. ¿Cuánto es el baseline B = |T| en mm? *(Norma euclidiana: sqrt(Tx² + Ty² + Tz²))*
2. ¿El resultado es razonable para dos ESP32-XIAO montadas juntas?
3. Si el baseline fuera B = 0.001m (1mm), ¿qué problema habría con la medición de profundidad? *(Pista: volvé a la fórmula Z = f*B/d)*
4. ¿Y si el baseline fuera B = 1.0m (1 metro)? ¿Qué ventaja y qué desventaja tendría?

---

# MÓDULO 2 — Lo que sigue después (mapa de ruta)

Este README cubre Módulo 0 (streaming) y el marco teórico completo del Módulo 1 (calibración).

Los siguientes pasos en orden son:

```
[Ya hecho]  Módulo 0: Streaming de ambas cámaras con threads
[Este doc]  Módulo 1: Teoría y captura para calibración
[Siguiente] Módulo 1B: Código de calibración (captura + calibrateCamera + stereoCalibrate)
[Después]   Módulo 2: Algoritmo SGBM + preprocesamiento CLAHE
[Después]   Módulo 3: Filtro WLS + estabilización Kalman/Media Móvil
[Final]     Módulo 4: Efecto AR reactivo a la distancia
```

---

## Recursos para profundizar (por módulo)

### Para el Módulo 1 (Calibración):
- 📖 **Libro:** Bradski & Kaehler, "Learning OpenCV 3" — Capítulo 18 (Camera Calibration)
- 📖 **Paper original:** Zhang, "A flexible new technique for camera calibration" (IEEE 2000) — [busca en Google Scholar]
- 🎥 **Video:** First Principles of Computer Vision (YouTube) — "Camera Calibration"
- 📚 **Docs OpenCV:** https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

### Para el Modelo Pinhole y álgebra:
- 📖 Hartley & Zisserman, "Multiple View Geometry in Computer Vision" — Capítulos 1-7
- 🎥 Cyrill Stachniss en YouTube: "Camera Models" (Universidad de Bonn)

### Para geometría estéreo:
- 📖 Szeliski, "Computer Vision: Algorithms and Applications" — Capítulo 7 (libre en su web)
- 🔗 https://learnopencv.com/stereo-vision-and-depth-estimation-using-opencv-ai-kit/

### Para CLAHE y histogramas:
- 📖 González & Woods, "Digital Image Processing" — Capítulo 3
- 🔗 https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

---

## Checklist antes de pasar al código de calibración

Deberías poder responder esto sin mirar el documento:

- [ ] ¿Qué son los marcadores JPEG SOI y EOI en hexadecimal?
- [ ] ¿Por qué dividimos X e Y por Z en la proyección perspectiva?
- [ ] ¿Qué representan fx, fy, cx, cy en la matriz K?
- [ ] ¿Qué es la disparidad y cómo se relaciona con la profundidad?
- [ ] ¿Por qué un baseline pequeño es problemático para medir objetos lejanos?
- [ ] ¿Por qué el tablero de ajedrez funciona para calibrar?
- [ ] ¿Qué es el error de reproyección y qué valor es aceptable?
- [ ] ¿Por qué necesitás mínimo 20-30 imágenes del tablero en diferentes posiciones?
- [ ] ¿Qué contiene el archivo YAML de calibración?
- [ ] ¿Cómo pasa OpenCV de un mapa de disparidad a profundidad en metros?

---

*Proyecto: Visión Artificial — Universidad Politécnica Salesiana*
*Período Lectivo: Abril – Agosto 2026*
*Docente: Ing. Vladimir Robles Bykbaev*
