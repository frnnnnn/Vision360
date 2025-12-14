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
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:053379869190:alertas-intrusos")
CAMERA_ID     = os.getenv("CAMERA_ID", "cam01")

# Frecuencias/umbrales
INFER_EVERY_SEC      = 1.0
TARGET_WIDTH         = 640
JPEG_QUALITY         = 70
MIN_CONFIDENCE       = 70
UPLOAD_ONLY_IF_PERSON= True

# Anti-spam
EVENT_COOLDOWN_SEC   = 12     # máx. 1 evento cada 12s
MIN_PERSON_FRAMES    = 2      # persona en 2 inferencias seguidas
EMAIL_COOLDOWN_SEC   = 180    # máx. 1 correo cada 3 min

# Labels
TOP_LABELS           = 4      # guarda las 4 más confiables

# Face Recognition
FACE_COLLECTION_ID   = os.getenv("FACE_COLLECTION_ID", "vision360-faces")
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "80"))
# ============================================

# Session / clients
session = boto3.session.Session(region_name=REGION)
rek = session.client("rekognition",
                     config=Config(read_timeout=5, connect_timeout=3, retries={"max_attempts": 2}))
s3  = session.client("s3")
ddb = session.resource("dynamodb").Table(DDB_TABLE)
sns = session.client("sns")

import socket
from urllib.parse import urlparse

def is_reachable(url, timeout=3):
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (80 if parsed.scheme == 'http' else 443)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

# Cámara
camera_url = os.getenv("CAMERA_URL")
# If camera_url is a number (e.g. "0"), convert to int
if camera_url and camera_url.isdigit():
    camera_url = int(camera_url)
elif not camera_url:
    camera_url = 0

# Set a timeout for the camera connection (5 seconds) - works for FFMPEG backend
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"

print("STARTING CAMERA SCRIPT", flush=True)
print(f"Connecting to camera: {camera_url}", flush=True)

use_webcam = False
# Check IP camera reachability
if isinstance(camera_url, str) and not is_reachable(camera_url):
    print(f"Camera at {camera_url} is unreachable. Falling back to local webcam (0)...", flush=True)
    use_webcam = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    # Try opening IP camera
    cap = cv2.VideoCapture(camera_url)

# If IP camera failed to open, fallback to local
if not cap.isOpened() and not use_webcam and camera_url != 0:
    print(f"Failed to open camera at {camera_url}. Falling back to local webcam (0)...", flush=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    use_webcam = True

# Verify we can read a frame
if cap.isOpened():
    # Try to read one frame to ensure it works
    ret, frame = cap.read()
    if not ret:
        print(f"Opened camera but failed to read frame. Falling back to local webcam (0)...", flush=True)
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        use_webcam = True
        
        # Check again after fallback
        if cap.isOpened():
             ret, frame = cap.read()
             if not ret:
                 print("Failed to read frame from local webcam too.", flush=True)
    else:
        print("Camera check passed: Frame read successfully.", flush=True)

if not cap.isOpened():
    print("Error: Could not open any camera.", flush=True)
    raise RuntimeError("No se pudo abrir la webcam.")
else:
    print(f"Camera opened successfully. Using {'Webcam (0)' if use_webcam else camera_url}", flush=True)

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
    s3.put_object(Bucket=S3_BUCKET, Key=img_key, Body=buf.tobytes(), ContentType="image/jpeg")
    return img_key

def write_event_ddb(item):
    ddb.put_item(Item=item)

def search_face_in_collection(image_bytes):
    """Search for a face in the Rekognition collection."""
    try:
        response = rek.search_faces_by_image(
            CollectionId=FACE_COLLECTION_ID,
            Image={"Bytes": image_bytes},
            FaceMatchThreshold=FACE_MATCH_THRESHOLD,
            MaxFaces=1
        )
        if response.get("FaceMatches"):
            face_match = response["FaceMatches"][0]
            face_id = face_match["Face"]["FaceId"]
            similarity = face_match["Similarity"]
            # Get person name from DynamoDB
            try:
                person_resp = ddb.get_item(Key={"event_id": f"FACE#{face_id}"})
                person_name = person_resp.get("Item", {}).get("name", "Unknown")
            except Exception:
                person_name = "Unknown"
            return {
                "match": True,
                "face_id": face_id,
                "name": person_name,
                "similarity": similarity
            }
        return {"match": False}
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidParameterException':
            return {"match": False, "message": "No face detected"}
        print(f"Face search error: {e}")
        return {"match": False}
    except Exception as e:
        print(f"Face search error: {e}")
        return {"match": False}

def publish_alert_sns(item):
    person_name = item.get('person_name', 'Desconocido')
    if item.get('authorized'):
        msg = (f"[INFO] Persona AUTORIZADA detectada\n"
               f"Nombre: {person_name}\n"
               f"cam: {item['camera_id']} conf: {float(item.get('confidence',0)):.1f}%\n"
               f"event_id: {item['event_id']}")
        subject = "Acceso autorizado"
    else:
        msg = (f"[ALERTA] Persona NO autorizada detectada\n"
               f"cam: {item['camera_id']} conf: {float(item.get('confidence',0)):.1f}%\n"
               f"frame: s3://{S3_BUCKET}/{item.get('s3_key','')}\n"
               f"event_id: {item['event_id']}")
        subject = "Alerta intrusión"
    sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=msg)

def run_inference(frame_bgr):
    global last_boxes, last_labels_text, person_flag, infer_in_flight
    global last_event_ts, person_streak, prev_person_state, last_email_ts

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

        # Face recognition - search if person is authorized
        authorized = False
        person_name = ""
        face_id = ""
        similarity = 0.0
        
        if found_person:
            face_result = search_face_in_collection(buf.tobytes())
            if face_result.get("match"):
                authorized = True
                person_name = face_result.get("name", "Unknown")
                face_id = face_result.get("face_id", "")
                similarity = face_result.get("similarity", 0.0)
                print(f"✓ AUTHORIZED: {person_name} (similarity: {similarity:.1f}%)")
            else:
                print(f"✗ UNAUTHORIZED: Unknown person detected")

        result_item = {
            "event_id": event_id,
            "timestamp": ts_ms,                       # int OK
            "camera_id": CAMERA_ID,
            "person_detected": found_person,          # bool OK
            "authorized": authorized,
            "person_name": person_name,
            "face_id": face_id,
            "face_similarity": Decimal(str(round(similarity, 1))) if similarity else Decimal("0"),
            "confidence": Decimal(str(round(best_conf, 1))),
            "labels": compact_labels,
            "s3_key": ""
        }

        # Cooldown de eventos (S3+DDB)
        now = time.time()
        should_record = ((not UPLOAD_ONLY_IF_PERSON) or found_person) and (now - last_event_ts >= EVENT_COOLDOWN_SEC)
        if not should_record:
            return

        last_event_ts = now

        # Subir thumbnail a S3 y persistir
        if (not UPLOAD_ONLY_IF_PERSON) or found_person:
            img_key = upload_to_s3(frame_infer, event_id, date_str)
            if img_key:
                result_item["s3_key"] = img_key

            write_event_ddb(result_item)

            # Envío de correo (solo al entrar en estado o cada EMAIL_COOLDOWN_SEC)
            entered_alert_state = (found_person and not result_item["authorized"] and not prev_person_state)
            email_cooldown_ok = (time.time() - last_email_ts) >= EMAIL_COOLDOWN_SEC
            if (entered_alert_state or email_cooldown_ok) and found_person and not result_item["authorized"] and SNS_TOPIC_ARN:
                publish_alert_sns(result_item)
                last_email_ts = time.time()

            prev_person_state = found_person

    except (BotoCoreError, ClientError) as e:
        with infer_lock:
            response = getattr(e, "response", None) or {}
            error = response.get("Error", {})
            code = error.get("Code", type(e).__name__)
            last_labels_text.appendleft(f"AWSERR:{code}")
            print(f"AWS Error: {e}")
    except Exception as e:
        with infer_lock:
            last_labels_text.appendleft(f"ERR:{type(e).__name__}")
    finally:
        with infer_lock:
            infer_in_flight = False

def main():
    global last_infer_t, infer_in_flight
    cv2.namedWindow("Detección en vivo (Plan B)", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
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
            hud = "Inferencia: EN CURSO" if infer_in_flight else "Inferencia: IDLE"

        # Dibujo de overlay
        for (lx1, ly1, lx2, ly2) in boxes:
            x1 = int(lx1 * w); y1 = int(ly1 * h)
            x2 = int(lx2 * w); y2 = int(ly2 * h)
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(disp, "Person", (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        y = 20
        for txt in labels_to_draw[:5]:
            cv2.putText(disp, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            y += 22

        cv2.putText(disp, hud, (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Detección en vivo (Plan B)", disp)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
