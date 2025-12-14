import os, time, json, threading
from collections import deque
from datetime import datetime
from decimal import Decimal

import cv2
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================== CONFIG ==================
REGION        = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET     = os.getenv("S3_BUCKET", "valdivia-deteccion-prototipo")
S3_PREFIX     = "events/raw"
DDB_TABLE     = os.getenv("DDB_TABLE", "deteccion_eventos")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "")
CAMERA_ID     = os.getenv("CAMERA_ID", "cam01")

# Frecuencias/umbrales
INFER_EVERY_SEC      = 1.0
TARGET_WIDTH         = 640
JPEG_QUALITY         = 70
MIN_CONFIDENCE       = 70
UPLOAD_ONLY_IF_PERSON= True
HEARTBEAT_SEC        = 5      # Check config every 5s for faster toggle response

# Anti-spam
EVENT_COOLDOWN_SEC   = 12     # máx. 1 evento cada 12s
MIN_PERSON_FRAMES    = 2      # persona en 2 inferencias seguidas
EMAIL_COOLDOWN_SEC   = 180    # máx. 1 correo cada 3 min

# Labels
TOP_LABELS           = 4      # guarda las 4 más confiables
# ============================================

# Session / clients
session = boto3.session.Session(region_name=REGION)
rek = session.client("rekognition",
                     config=Config(read_timeout=5, connect_timeout=3, retries={"max_attempts": 2}))
s3  = session.client("s3")
ddb = session.resource("dynamodb").Table(DDB_TABLE)
sns = session.client("sns")

# Cámara (Inicialización diferida en main)
cap = None

# Estado UI
last_boxes = []
last_labels_text = deque(maxlen=6)
person_flag = False

# Estado de control (concurrency)
infer_lock = threading.Lock()
infer_in_flight = False
last_infer_t = 0.0

# Estado anti-spam (globales)
last_event_ts = 0.0
person_streak = 0
prev_person_state = False
last_email_ts = 0.0
last_heartbeat_ts = 0.0

# Cache de configuración
camera_name = CAMERA_ID
camera_location = "Unknown"
camera_url = ""
camera_user = ""
camera_pass = ""

def get_camera_config():
    """Obtiene nombre, ubicación y URL desde DynamoDB."""
    global camera_name, camera_location, camera_url, camera_user, camera_pass
    try:
        resp = ddb.get_item(Key={"event_id": f"CONFIG#{CAMERA_ID}"})
        item = resp.get("Item")
        if item:
            camera_name = item.get("name", CAMERA_ID)
            camera_location = item.get("location", "Unknown")
            camera_url = item.get("url", "")
            camera_user = item.get("username", "")
            camera_pass = item.get("password", "")
            is_active = item.get("is_active", True)
            print(f"DEBUG DDB ITEM: {item}")
            print(f"Config loaded: {camera_name} | Active: {is_active} | URL: {camera_url}")
            return is_active
    except Exception as e:
        print(f"Error loading config: {e}")
        # Fallback to env vars
        print("Intentando cargar configuración desde variables de entorno...")
        camera_url = os.getenv("CAMERA_URL", "")
        camera_user = os.getenv("CAMERA_USER", "")
        camera_pass = os.getenv("CAMERA_PASS", "")
        print(f"Fallback Config: URL: {camera_url}")
    return True

# WebSocket Global
ws_client = None
WS_URL = f"ws://localhost:8000/ws/camera/{CAMERA_ID}"

def connect_ws():
    global ws_client
    try:
        import websocket
        ws_client = websocket.create_connection(WS_URL)
        print("WebSocket Connected!")
    except Exception as e:
        print(f"WS Connection Failed: {e}")
        ws_client = None

def upload_live_preview(frame):
    """Sube frame via WebSocket para streaming real-time."""
    global ws_client
    try:
        # Resize for performance (e.g. 640px width)
        h, w = frame.shape[:2]
        new_h = int(h * 640 / w)
        small = cv2.resize(frame, (640, new_h))
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        
        if ok and ws_client:
            try:
                ws_client.send_binary(buf.tobytes())
            except Exception:
                # Reconnect on failure
                print("WS Lost. Reconnecting...")
                connect_ws()
        elif not ws_client:
            # Try to connect occasionally
            if int(time.time()) % 5 == 0:
                connect_ws()
                
    except Exception as e:
        pass

# Initialize WebSocket connection at startup
print(f"Connecting WebSocket to {WS_URL}...")
connect_ws()

def send_heartbeat():
    """Envía latido a DynamoDB para indicar estado Online."""
    try:
        ddb.update_item(
            Key={"event_id": f"CONFIG#{CAMERA_ID}"},
            UpdateExpression="SET last_heartbeat = :t, #st = :s, camera_id = :c",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={
                ":t": int(time.time()),
                ":s": "ONLINE",
                ":c": CAMERA_ID
            }
        )
    except Exception as e:
        print(f"Heartbeat error: {e}")

def upload_to_s3(frame_bgr, event_id, date_str):
    """Sube thumbnail (ancho 480) a S3 para ahorrar peso."""
    h, w = frame_bgr.shape[:2]
    if w > 480:
        new_h = int(h * 480 / w)
        frame_bgr = cv2.resize(frame_bgr, (480, new_h))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if not ok:
        return None
    img_key = f"{S3_PREFIX}/{date_str}/{event_id}.jpg"
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=img_key, Body=buf.tobytes(), ContentType="image/jpeg")
        return img_key
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None

def write_event_ddb(item):
    try:
        ddb.put_item(Item=item)
    except Exception as e:
        print(f"Error writing to DynamoDB: {e}")
        # Fallback to backend API
        try:
            import requests
            # Convert Decimal to float/int for JSON serialization
            import json
            from decimal import Decimal
            
            def decimal_default(obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                raise TypeError
            
            # Prepare item for JSON
            item_json = json.loads(json.dumps(item, default=decimal_default))
            
            print("Sending event to backend API...")
            requests.post("http://localhost:8000/events", json=item_json)
        except Exception as api_e:
            print(f"Error sending to backend API: {api_e}")

def publish_alert_sns(item):
    if not SNS_TOPIC_ARN:
        return
    
    loc_str = f"{camera_name} ({camera_location})"
    
    # Generate presigned URL for image (valid for 24 hours)
    img_url = "No disponible"
    if item.get("s3_key"):
        try:
            img_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': item.get("s3_key")},
                ExpiresIn=86400  # 24 horas
            )
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
    
    msg = (f"🚨 [ALERTA VISION360] Persona NO autorizada\n\n"
           f"📍 Ubicación: {loc_str}\n"
           f"🎯 Confianza: {float(item.get('confidence',0)):.1f}%\n"
           f"⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
           f"📸 Ver imagen del intruso (Click para abrir):\n{img_url}\n\n"
           f"💾 ID Evento: {item['event_id']}")
    
    try:
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=f"🚨 Alerta Vision360: {loc_str}", Message=msg)
        with open("debug.log", "a") as f:
            f.write(f"{datetime.now()} - Alerta SNS enviada EXITOSAMENTE.\n")
    except Exception as e:
        with open("debug.log", "a") as f:
            f.write(f"{datetime.now()} - Error sending SNS: {e}\n")

def run_inference(frame_bgr):
    global last_boxes, last_labels_text, person_flag, infer_in_flight
    global last_event_ts, person_streak, prev_person_state, last_email_ts, last_heartbeat_ts

    # Heartbeat check & Config Refresh
    now = time.time()
    if now - last_heartbeat_ts > HEARTBEAT_SEC:
        active = get_camera_config()
        send_heartbeat()
        last_heartbeat_ts = now
        if not active:
            print("Cámara INACTIVA por configuración.")
            return

    try:
        # Redimensiona para bajar latencia de envío a Rekognition
        h, w = frame_bgr.shape[:2]
        scale = TARGET_WIDTH / max(1, w)
        frame_infer = cv2.resize(frame_bgr, (int(w*scale), int(h*scale))) if scale < 1.0 else frame_bgr

        ok, buf = cv2.imencode(".jpg", frame_infer, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return

        resp = rek.detect_labels(Image={"Bytes": buf.tobytes()}, MaxLabels=15, MinConfidence=MIN_CONFIDENCE)

        # Parseo de labels y cajas
        new_boxes, new_texts = [], []
        found_person, best_conf = False, 0.0

        # Ordena labels por confianza y limita
        labels_sorted = sorted(resp.get("Labels", []),
                               key=lambda x: x.get("Confidence", 0), reverse=True)[:TOP_LABELS]

        for lab in labels_sorted:
            new_texts.append(f"{lab['Name']} {lab['Confidence']:.1f}%")
            if lab["Name"].lower() == "person":
                found_person = True
                best_conf = max(best_conf, lab["Confidence"])
                for inst in lab.get("Instances", []):
                    bb = inst.get("BoundingBox", {})
                    if all(k in bb for k in ("Width","Height","Left","Top")):
                        new_boxes.append((bb["Left"], bb["Top"],
                                          bb["Left"]+bb["Width"], bb["Top"]+bb["Height"]))

        # “N frames seguidos” con persona
        person_streak = person_streak + 1 if found_person else 0
        if found_person and person_streak < MIN_PERSON_FRAMES:
            # Aún no dispares eventos; solo actualiza overlay
            with infer_lock:
                last_boxes = new_boxes
                last_labels_text.clear()
                last_labels_text.extend(new_texts)
                person_flag = found_person
            return

        # Actualiza overlay siempre
        with infer_lock:
            last_boxes = new_boxes
            last_labels_text.clear()
            last_labels_text.extend(new_texts)
            person_flag = found_person

        # Arma item (floats -> Decimal)
        ts_ms = int(time.time() * 1000)
        event_id = f"{CAMERA_ID}-{ts_ms}"
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

        compact_labels = []
        for lab in labels_sorted:
            compact_labels.append({
                "Name": lab.get("Name"),
                "Confidence": Decimal(str(round(lab.get("Confidence", 0.0), 1)))
            })

        result_item = {
            "event_id": event_id,
            "timestamp": ts_ms,                       # int OK
            "camera_id": CAMERA_ID,
            "person_detected": found_person,          # bool OK
            "authorized": False,                      # demo: todo NO autorizado
            "confidence": Decimal(str(round(best_conf, 1))),
            "labels": compact_labels,
            "s3_key": ""
        }

        # Cooldown de eventos (S3+DDB)
        should_record = ((not UPLOAD_ONLY_IF_PERSON) or found_person) and (now - last_event_ts >= EVENT_COOLDOWN_SEC)
        if not should_record:
            return

        last_event_ts = now

        # Subir thumbnail a S3 y persistir
        if (not UPLOAD_ONLY_IF_PERSON) or found_person:
            print(f"Registrando evento: {event_id}")
            img_key = upload_to_s3(frame_infer, event_id, date_str)
            if img_key:
                result_item["s3_key"] = img_key

            write_event_ddb(result_item)

            # Envío de correo (solo al entrar en estado o cada EMAIL_COOLDOWN_SEC)
            entered_alert_state = (found_person and not result_item["authorized"] and not prev_person_state)
            email_cooldown_ok = (time.time() - last_email_ts) >= EMAIL_COOLDOWN_SEC
            
            with open("debug.log", "a") as f:
                if found_person:
                    f.write(f"{datetime.now()} - Persona detectada. AlertState: {entered_alert_state}, Cooldown: {email_cooldown_ok}, ARN: {bool(SNS_TOPIC_ARN)}\n")

            if (entered_alert_state or email_cooldown_ok) and found_person and not result_item["authorized"] and SNS_TOPIC_ARN:
                with open("debug.log", "a") as f:
                    f.write(f"{datetime.now()} - Intentando enviar alerta SNS a {SNS_TOPIC_ARN}...\n")
                publish_alert_sns(result_item)
                last_email_ts = time.time()

            prev_person_state = found_person

    except (BotoCoreError, ClientError) as e:
        with infer_lock:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
            last_labels_text.appendleft(f"AWSERR:{code}")
    except Exception as e:
        with infer_lock:
            last_labels_text.appendleft(f"ERR:{type(e).__name__}")
    finally:
        with infer_lock:
            infer_in_flight = False

def main():
    global last_infer_t, infer_in_flight, cap
    
    # Cargar config inicial
    get_camera_config()
    
    # Inicializar cámara
    src = 0
    if camera_url and camera_url.startswith("http"):
        src = camera_url
        # Inject credentials if available
        if camera_user and camera_pass:
            try:
                protocol, address = camera_url.split("://", 1)
                src = f"{protocol}://{camera_user}:{camera_pass}@{address}"
                print(f"Usando credenciales para conectar a {address}")
            except Exception:
                pass
        print(f"Conectando a cámara IP: {src}")
    else:
        print("Usando webcam USB local")
        
    # Inicializar cámara con reintentos y fallback
    while True:
        # 1. Intentar URL original
        current_src = src
        cap = cv2.VideoCapture(current_src)
        
        # 2. Si falla y es HTTP, intentar appending /video
        if not cap.isOpened() and isinstance(src, str) and src.startswith("http") and not src.endswith("/video"):
            print(f"Fallo conexión a {src}, intentando con /video...")
            current_src = f"{src}/video" if not src.endswith("/") else f"{src}video"
            cap = cv2.VideoCapture(current_src)

        if cap.isOpened():
            print(f"✅ Cámara conectada exitosamente: {current_src}")
            break
        
        print(f"❌ Error: No se pudo conectar a {src}. Reintentando en 5s...")
        time.sleep(5)
        # Recargar config por si el usuario la cambia
        get_camera_config()
        if camera_url and camera_url.startswith("http"):
            src = camera_url
            if camera_user and camera_pass:
                try:
                    protocol, address = camera_url.split("://", 1)
                    src = f"{protocol}://{camera_user}:{camera_pass}@{address}"
                except: pass

    cv2.namedWindow("Vision360 - Detección en vivo", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost stream or failed to grab frame.")
            cap.release()
            
            # Si fallamos inmediatamente (o casi), intentar fallback si no lo hemos hecho
            if isinstance(src, str) and src.startswith("http") and not src.endswith("/video") and "video" not in src:
                 print(f"Posible URL incorrecta. Intentando fallback a {src}/video ...")
                 src = f"{src}/video" if not src.endswith("/") else f"{src}video"
                 current_src = src
            
            time.sleep(2)
            cap = cv2.VideoCapture(current_src)
            continue
            
        frame = cv2.flip(frame, 1)
        
        # Check config every 5 seconds for faster toggle response
        now = time.time()
        global last_heartbeat_ts
        if now - last_heartbeat_ts > HEARTBEAT_SEC:
            active = get_camera_config()
            send_heartbeat()
            last_heartbeat_ts = now
            if not active:
                # When inactive, just show local preview but don't stream or infer
                cv2.imshow("Vision360 - Detección en vivo (PAUSADO)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
        
        # Live Streaming - send every frame for smooth 30fps video
        upload_live_preview(frame)
        
        disp = frame.copy()
        h, w = disp.shape[:2]

        now = time.time()
        with infer_lock:
            can_launch = (now - last_infer_t >= INFER_EVERY_SEC) and (infer_in_flight is False)
        if can_launch:
            last_infer_t = now
            with infer_lock:
                infer_in_flight = True
            threading.Thread(target=run_inference, args=(frame.copy(),), daemon=True).start()

        with infer_lock:
            boxes = list(last_boxes)
            labels_to_draw = list(last_labels_text)
            flag_person = person_flag
            hud = f"Vision360: {camera_name} ({camera_location})"

        # Dibujo de overlay
        for (lx1, ly1, lx2, ly2) in boxes:
            x1 = int(lx1 * w); y1 = int(ly1 * h)
            x2 = int(lx2 * w); y2 = int(ly2 * h)
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,255), 2) # Rojo para alerta
            cv2.putText(disp, "PERSONA DETECTADA", (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

        y = 20
        for txt in labels_to_draw[:5]:
            cv2.putText(disp, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            y += 22

        cv2.putText(disp, hud, (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Vision360 - Detección en vivo", disp)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
