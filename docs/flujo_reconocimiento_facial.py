"""
================================================================================
VISION360 - FLUJO COMPLETO DE RECONOCIMIENTO FACIAL
================================================================================
Este archivo documenta el flujo completo desde captura de imagen hasta
clasificación como "AUTORIZADO" o "INTRUSO".

Autor: Vision360 Team
Versión: 1.0
================================================================================
"""

import boto3
import base64
import uuid
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

AWS_REGION = "us-east-1"
FACE_COLLECTION_ID = "vision360-faces"      # Colección de rostros en Rekognition
FACE_MATCH_THRESHOLD = 80.0                  # Umbral de similitud (80%)
DYNAMODB_TABLE = "vision360-events"          # Tabla de eventos y personas

# Clientes AWS
rekognition = boto3.client("rekognition", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMODB_TABLE)


# ============================================================================
# PASO 1: REGISTRAR PERSONA (Una sola vez por cada persona autorizada)
# ============================================================================

def registrar_persona(imagen_bytes: bytes, nombre: str) -> dict:
    """
    Registra una nueva persona en el sistema.
    
    Args:
        imagen_bytes: Imagen de la cara en formato bytes (JPEG/PNG)
        nombre: Nombre de la persona (ej: "Juan Pérez")
    
    Returns:
        dict con face_id, person_id y resultado
    
    Proceso:
        1. Envía imagen a AWS Rekognition para indexar la cara
        2. Rekognition genera un face_id único
        3. Guardamos face_id + nombre en DynamoDB
    """
    person_id = str(uuid.uuid4())  # ID único para la persona
    
    # ---------- LLAMADA A AWS REKOGNITION ----------
    response = rekognition.index_faces(
        CollectionId=FACE_COLLECTION_ID,
        Image={"Bytes": imagen_bytes},
        ExternalImageId=person_id,           # Asociamos con person_id
        DetectionAttributes=["ALL"],
        MaxFaces=1,
        QualityFilter="AUTO"
    )
    
    # Extraer face_id generado por Rekognition
    if not response.get("FaceRecords"):
        return {"success": False, "error": "No se detectó rostro en la imagen"}
    
    face_id = response["FaceRecords"][0]["Face"]["FaceId"]
    
    # ---------- GUARDAR EN DYNAMODB ----------
    # Entrada de persona (agrupa múltiples fotos)
    dynamodb.put_item(Item={
        "event_id": f"PERSON#{person_id}",   # Clave primaria
        "timestamp": 0,
        "person_id": person_id,
        "name": nombre,
        "face_ids": [face_id]                 # Lista de caras registradas
    })
    
    # Mapeo face_id -> person_id (para búsquedas rápidas)
    dynamodb.put_item(Item={
        "event_id": f"FACE#{face_id}",
        "timestamp": 0,
        "face_id": face_id,
        "person_id": person_id,
        "name": nombre
    })
    
    return {
        "success": True,
        "face_id": face_id,
        "person_id": person_id,
        "name": nombre,
        "message": f"Persona '{nombre}' registrada exitosamente"
    }


# ============================================================================
# PASO 2: BUSCAR ROSTRO EN COLECCIÓN (Cada vez que se detecta una cara)
# ============================================================================

def buscar_rostro(imagen_bytes: bytes) -> dict:
    """
    Busca si el rostro en la imagen está registrado en el sistema.
    
    Args:
        imagen_bytes: Imagen capturada de la cámara
    
    Returns:
        dict con:
        - match: True si encontró coincidencia, False si es desconocido
        - authorized: True si está autorizado (igual que match)
        - name: Nombre de la persona (solo si match=True)
        - similarity: Porcentaje de similitud (0-100)
    
    Este es el CORE del sistema de reconocimiento.
    """
    try:
        # ---------- BÚSQUEDA EN AWS REKOGNITION ----------
        response = rekognition.search_faces_by_image(
            CollectionId=FACE_COLLECTION_ID,
            Image={"Bytes": imagen_bytes},
            FaceMatchThreshold=FACE_MATCH_THRESHOLD,  # Solo matches >= 80%
            MaxFaces=1                                 # Solo el mejor match
        )
        
        # ---------- EVALUAR RESULTADO ----------
        if response.get("FaceMatches"):
            # ✅ SE ENCONTRÓ COINCIDENCIA -> PERSONA AUTORIZADA
            face_match = response["FaceMatches"][0]
            face_id = face_match["Face"]["FaceId"]
            similarity = face_match["Similarity"]
            
            # Obtener nombre de DynamoDB
            person_data = dynamodb.get_item(
                Key={"event_id": f"FACE#{face_id}"}
            ).get("Item", {})
            
            return {
                "match": True,                              # ⬅️ COINCIDENCIA
                "authorized": True,                         # ⬅️ AUTORIZADO
                "name": person_data.get("name", "Unknown"),
                "face_id": face_id,
                "person_id": person_data.get("person_id"),
                "similarity": round(similarity, 2)
            }
        else:
            # ❌ NO HAY COINCIDENCIA -> INTRUSO
            return {
                "match": False,                             # ⬅️ SIN COINCIDENCIA
                "authorized": False,                        # ⬅️ NO AUTORIZADO = INTRUSO
                "message": "Persona no registrada en el sistema"
            }
            
    except rekognition.exceptions.InvalidParameterException:
        # No se detectó cara en la imagen
        return {
            "match": False,
            "authorized": False,
            "person_detected": False,
            "message": "No se detectó rostro en la imagen"
        }


# ============================================================================
# PASO 3: PROCESAR FRAME DE CÁMARA (Loop principal del sistema)
# ============================================================================

def procesar_frame_camara(imagen_bytes: bytes, camera_id: str) -> dict:
    """
    Procesa un frame capturado de la cámara.
    Este es el punto de entrada principal del sistema de vigilancia.
    
    Args:
        imagen_bytes: Frame capturado de la cámara
        camera_id: Identificador de la cámara (ej: "cam01")
    
    Returns:
        dict con evento completo para mostrar en dashboard
    
    Flujo:
        1. Detectar si hay persona en el frame
        2. Si hay persona, buscar en colección de autorizados
        3. Clasificar como AUTORIZADO o INTRUSO
        4. Guardar evento en DynamoDB
        5. Retornar para mostrar en frontend
    """
    timestamp = int(datetime.now().timestamp() * 1000)
    event_id = f"{camera_id}-{timestamp}"
    
    # ---------- DETECTAR Y BUSCAR PERSONA ----------
    resultado = buscar_rostro(imagen_bytes)
    
    # ---------- CONSTRUIR EVENTO ----------
    evento = {
        "event_id": event_id,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "person_detected": resultado.get("person_detected", True),
        
        # ⬇️ ESTA ES LA CLASIFICACIÓN CLAVE ⬇️
        "authorized": resultado.get("authorized", False),
        # authorized=True  -> Persona conocida (VERDE en dashboard)
        # authorized=False -> INTRUSO (ROJO en dashboard)
        
        "person_name": resultado.get("name"),
        "face_id": resultado.get("face_id"),
        "face_similarity": resultado.get("similarity"),
        "confidence": resultado.get("similarity", 0)
    }
    
    # ---------- GUARDAR EN DYNAMODB ----------
    dynamodb.put_item(Item=evento)
    
    return evento


# ============================================================================
# PASO 4: CLASIFICACIÓN PARA EL FRONTEND
# ============================================================================

def clasificar_evento(evento: dict) -> str:
    """
    Clasifica un evento para mostrar en el dashboard.
    
    Esta función traduce los campos técnicos a categorías
    legibles para el usuario.
    
    Args:
        evento: Diccionario con person_detected y authorized
    
    Returns:
        "AUTORIZADO" | "INTRUSO" | "MOVIMIENTO"
    """
    person_detected = evento.get("person_detected", False)
    authorized = evento.get("authorized", False)
    
    if person_detected and authorized:
        # ✅ Cara detectada + está en sistema = AUTORIZADO
        return "AUTORIZADO"
    
    elif person_detected and not authorized:
        # ❌ Cara detectada + NO está en sistema = INTRUSO
        return "INTRUSO"
    
    else:
        # ⚪ No se detectó cara = solo movimiento
        return "MOVIMIENTO"


def clasificar_severidad(evento: dict) -> str:
    """
    Clasifica la severidad de un evento para el panel de alertas.
    
    Args:
        evento: Diccionario con person_detected, authorized, confidence
    
    Returns:
        "ALTA" | "MEDIA" | "BAJA"
    """
    person_detected = evento.get("person_detected", False)
    authorized = evento.get("authorized", False)
    confidence = evento.get("confidence", 100)
    
    if person_detected and not authorized:
        # 🔴 INTRUSO = Severidad ALTA (requiere atención inmediata)
        return "ALTA"
    
    elif confidence and confidence < 70:
        # 🟡 Baja confianza = Severidad MEDIA (revisar manualmente)
        return "MEDIA"
    
    else:
        # 🔵 Normal = Severidad BAJA
        return "BAJA"


# ============================================================================
# EJEMPLO DE USO COMPLETO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de cómo usar el sistema completo.
    
    NOTA: Este código es ilustrativo. En producción:
    - Las imágenes vienen del cliente de cámara (main.py)
    - Los eventos se envían por WebSocket al frontend
    - El frontend muestra dashboard en tiempo real
    """
    
    # Simular imagen capturada (en producción viene de cv2.VideoCapture)
    with open("imagen_ejemplo.jpg", "rb") as f:
        imagen_bytes = f.read()
    
    # --------------------------------------------------
    # ESCENARIO 1: Registrar nueva persona autorizada
    # --------------------------------------------------
    resultado_registro = registrar_persona(
        imagen_bytes=imagen_bytes,
        nombre="Juan Pérez"
    )
    print(f"Registro: {resultado_registro}")
    # Output: {"success": True, "face_id": "abc-123", "name": "Juan Pérez"}
    
    # --------------------------------------------------
    # ESCENARIO 2: Procesar frame donde aparece Juan
    # --------------------------------------------------
    evento_autorizado = procesar_frame_camara(
        imagen_bytes=imagen_bytes,
        camera_id="cam01"
    )
    print(f"Evento: {clasificar_evento(evento_autorizado)}")
    # Output: "AUTORIZADO" (porque Juan está registrado)
    
    # --------------------------------------------------
    # ESCENARIO 3: Procesar frame de persona desconocida
    # --------------------------------------------------
    with open("imagen_desconocido.jpg", "rb") as f:
        imagen_intruso = f.read()
    
    evento_intruso = procesar_frame_camara(
        imagen_bytes=imagen_intruso,
        camera_id="cam01"
    )
    print(f"Evento: {clasificar_evento(evento_intruso)}")
    # Output: "INTRUSO" (porque NO está registrado)
    print(f"Severidad: {clasificar_severidad(evento_intruso)}")
    # Output: "ALTA" (intrusos siempre son severidad alta)


# ============================================================================
# DIAGRAMA DE FLUJO
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FLUJO DE RECONOCIMIENTO FACIAL                       │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │    CÁMARA    │
    │  (IP/RTSP)   │
    └──────┬───────┘
           │ frame JPEG
           ▼
    ┌──────────────┐
    │   CLIENTE    │  main.py captura frames cada 3 segundos
    │   (Python)   │
    └──────┬───────┘
           │ WebSocket + base64
           ▼
    ┌──────────────┐
    │   BACKEND    │  FastAPI recibe y procesa
    │   (FastAPI)  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                        AWS REKOGNITION                           │
    │  search_faces_by_image(CollectionId, Image, Threshold=80%)       │
    └───────────────────────────────┬──────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
              ▼                                           ▼
    ┌─────────────────────┐                    ┌─────────────────────┐
    │  FaceMatches: []    │                    │  FaceMatches: [...]  │
    │  (Lista vacía)      │                    │  similarity: 95.2%   │
    └──────────┬──────────┘                    └──────────┬───────────┘
               │                                          │
               ▼                                          ▼
    ┌─────────────────────┐                    ┌─────────────────────┐
    │   authorized: FALSE │                    │   authorized: TRUE  │
    │   ❌ INTRUSO        │                    │   ✅ AUTORIZADO     │
    │   Severidad: ALTA   │                    │   Severidad: BAJA   │
    └──────────┬──────────┘                    └──────────┬──────────┘
               │                                          │
               └───────────────────┬──────────────────────┘
                                   │
                                   ▼
                          ┌──────────────┐
                          │   DYNAMODB   │  Guardar evento
                          │   (Eventos)  │
                          └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │   FRONTEND   │  Mostrar en dashboard
                          │   (Next.js)  │  con color rojo/verde
                          └──────────────┘
"""
