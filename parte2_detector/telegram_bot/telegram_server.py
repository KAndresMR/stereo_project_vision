#!/Users/andresmorocho/miniconda3/envs/vision/bin/python
# ==============================================================================
# Servidor HTTP de Recepción y Procesamiento de Alertas (YOLO + Telegram)
# Recibe capturas y videos desde la aplicación C++, ejecuta inferencia de
# segmentación de instancias con YOLO y transmite los resultados al usuario.
# ==============================================================================
import os
import tempfile
import cv2
import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Configuración de credenciales del Bot de Telegram (seguridad por variables de entorno)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8897953514:AAEc5IcHn98ykv97BfvvyXGPUBclYe1S9sM")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1610725463")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Carga en memoria del modelo de segmentación de instancias YOLO preentrenado
print("[INFO] Inicializando modelo de segmentación YOLOv8n-seg...")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n-seg.pt')
model = YOLO(MODEL_PATH)
print("[OK] Modelo YOLO cargado exitosamente en memoria.")


def send_telegram_message(text):
    """Envía un mensaje de texto plano a la conversación de Telegram configurada."""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Falló el envío del mensaje de texto a Telegram: {e}")


def send_telegram_photo(photo_path, caption=""):
    """Adjunta y envía una fotografía por Telegram mediante una petición HTTP POST."""
    url = f"{TELEGRAM_API_URL}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo:
            payload = {"chat_id": CHAT_ID, "caption": caption}
            files = {"photo": photo}
            response = requests.post(url, data=payload, files=files, timeout=30)
            response.raise_for_status()
    except (requests.exceptions.RequestException, OSError) as e:
        print(f"[ERROR] Falló el envío de la fotografía a Telegram: {e}")


def send_telegram_video(video_path, caption=""):
    """Adjunta y envía un archivo de video por Telegram mediante una petición HTTP POST."""
    url = f"{TELEGRAM_API_URL}/sendVideo"
    try:
        with open(video_path, 'rb') as video:
            payload = {"chat_id": CHAT_ID, "caption": caption}
            files = {"video": video}
            response = requests.post(url, data=payload, files=files, timeout=60)
            response.raise_for_status()
    except (requests.exceptions.RequestException, OSError) as e:
        print(f"[ERROR] Falló el envío del video a Telegram: {e}")


def process_video_with_yolo(input_path, output_path):
    """
    Procesa un archivo de video aplicando la segmentación de instancias de YOLO frame a frame.
    Genera un nuevo archivo MP4 con las máscaras translúcidas superpuestas.
    """
    print("[INFO] Iniciando segmentación de video frame a frame con YOLO...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el flujo de video de entrada para segmentación.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Selección de códec H.264 compatible con la reproducción nativa de Telegram
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        # Respaldo con códec estándar si avc1 no está disponible
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia silenciosa para no saturar la salida estándar del servidor
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"[OK] Segmentación de video finalizada exitosamente ({frame_count} frames procesados).")
    return True


@app.route('/trigger', methods=['POST'])
def trigger():
    """Endpoint HTTP receptor de alertas disparadas por el cliente C++."""
    print("\n[HTTP] Petición entrante desde el sistema C++...")

    if 'photo' not in request.files or 'video' not in request.files:
        print("[ERROR] Petición rechazada: faltan flujos de imagen o video.")
        return jsonify({"error": "Falta archivo de fotografía o video"}), 400

    photo_file = request.files['photo']
    video_file = request.files['video']

    # Utilizar directorio temporal del sistema para evitar saturar el proyecto de artefactos
    temp_dir = tempfile.gettempdir()
    raw_photo_path = os.path.join(temp_dir, "raw_evidencia.jpg")
    raw_video_path = os.path.join(temp_dir, "raw_evidencia.mp4")
    seg_photo_path = os.path.join(temp_dir, "seg_evidencia.jpg")
    seg_video_path = os.path.join(temp_dir, "seg_evidencia.mp4")

    try:
        # Almacenamiento temporal de archivos recibidos
        photo_file.save(raw_photo_path)
        video_file.save(raw_video_path)

        # 1. Transmitir alerta inicial y fotografía original
        print("[TELEGRAM] Transmitiendo alerta e imagen original...")
        send_telegram_message("🚨 ALERTA: Vehículo objetivo [Bus Escolar] detectado en la escena.")
        send_telegram_photo(raw_photo_path, caption="📸 Imagen Original (Captura del sistema de detección C++)")

        # 2. Inferencia de segmentación en imagen clave
        print("[YOLO] Ejecutando segmentación de instancias sobre imagen clave...")
        results = model(raw_photo_path, verbose=False)
        segmented_img = results[0].plot()
        cv2.imwrite(seg_photo_path, segmented_img)

        # 3. Transmitir fotografía segmentada
        print("[TELEGRAM] Transmitiendo imagen segmentada...")
        send_telegram_photo(seg_photo_path, caption="🤖 Imagen Segmentada (Análisis de instancias YOLO)")

        # 4. Procesar y transmitir secuencia de video segmentada frame a frame
        if process_video_with_yolo(raw_video_path, seg_video_path):
            print("[TELEGRAM] Transmitiendo secuencia de video segmentada...")
            send_telegram_video(seg_video_path, caption="🎥 Secuencia de Video Segmentada (Inferencia frame a frame)")
        else:
            print("[WARNING] Falló la segmentación de video; transmitiendo video original como respaldo.")
            send_telegram_video(raw_video_path, caption="🎥 Secuencia de Video (Original)")

        print("[OK] Alerta procesada y notificaciones enviadas a Telegram satisfactoriamente.")
        return jsonify({"status": "ok"}), 200

    finally:
        # Limpieza de archivos temporales para mantener el sistema de archivos limpio
        for path in [raw_photo_path, raw_video_path, seg_photo_path, seg_video_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


if __name__ == '__main__':
    print("[INFO] Servidor Flask en escucha en puerto 5001 (0.0.0.0)...")
    app.run(host='0.0.0.0', port=5001, debug=False)
