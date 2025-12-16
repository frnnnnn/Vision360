from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import time
import asyncio
from aws_service import aws_service
from scanner_service import scanner_service
import cv2
import threading

app = FastAPI(title="Vision360 Enterprise API")

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        # active_connections: {camera_id: [WebSocket]} (viewers)
        self.viewers: Dict[str, List[WebSocket]] = {}
        # camera_connections: {camera_id: WebSocket} (streamers)
        self.cameras: Dict[str, WebSocket] = {}

    async def connect_viewer(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        if camera_id not in self.viewers:
            self.viewers[camera_id] = []
        self.viewers[camera_id].append(websocket)

    def disconnect_viewer(self, websocket: WebSocket, camera_id: str):
        if camera_id in self.viewers:
            if websocket in self.viewers[camera_id]:
                self.viewers[camera_id].remove(websocket)

    async def connect_camera(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        self.cameras[camera_id] = websocket

    def disconnect_camera(self, camera_id: str):
        if camera_id in self.cameras:
            del self.cameras[camera_id]

    async def broadcast_frame(self, camera_id: str, frame_data: bytes):
        if camera_id in self.viewers:
            for connection in self.viewers[camera_id]:
                try:
                    await connection.send_bytes(frame_data)
                except Exception:
                    pass # Handle broken pipes silently

manager = ConnectionManager()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Camera(BaseModel):
    camera_id: str
    name: str
    location: str
    description: Optional[str] = ""
    url: Optional[str] = ""
    username: Optional[str] = ""
    password: Optional[str] = ""
    status: Optional[str] = "OFFLINE"
    is_active: Optional[bool] = True
    live_preview: Optional[str] = ""
    last_heartbeat: Optional[float] = 0

class CameraUpdate(BaseModel):
    camera_id: Optional[str] = None
    name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    status: Optional[str] = None
    is_active: Optional[bool] = None
    live_preview: Optional[str] = None
    last_heartbeat: Optional[float] = None

@app.get("/")
def read_root():
    return {"status": "Vision360 API Online"}

@app.get("/cameras")
def get_cameras():
    return aws_service.get_cameras()

@app.post("/cameras")
def create_camera(cam: Camera):
    item = cam.dict()
    item["event_id"] = f"CONFIG#{cam.camera_id}"
    item["timestamp"] = 0
    if aws_service.save_camera(item):
        return {"status": "success", "camera": item}
    raise HTTPException(status_code=500, detail="Failed to save camera")

@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: str):
    if aws_service.delete_camera(cam_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=500, detail="Failed to delete camera")

@app.patch("/cameras/{cam_id}")
def update_camera(cam_id: str, cam: CameraUpdate):
    # Only update provided fields
    if aws_service.update_camera(cam_id, cam.dict(exclude_unset=True)):
        return {"status": "updated"}
    raise HTTPException(status_code=500, detail="Failed to update camera")

@app.get("/cameras/{cam_id}/live")
def get_live_url(cam_id: str):
    # Return WebSocket URL for the frontend
    return {"url": f"ws://localhost:8000/ws/view/{cam_id}"}

@app.websocket("/ws/camera/{camera_id}")
async def websocket_camera_endpoint(websocket: WebSocket, camera_id: str):
    await manager.connect_camera(websocket, camera_id)
    try:
        while True:
            data = await websocket.receive_bytes()
            await manager.broadcast_frame(camera_id, data)
    except WebSocketDisconnect:
        manager.disconnect_camera(camera_id)

@app.websocket("/ws/view/{camera_id}")
async def websocket_viewer_endpoint(websocket: WebSocket, camera_id: str):
    await manager.connect_viewer(websocket, camera_id)
    try:
        while True:
            # Keep connection alive, maybe receive control commands later
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_viewer(websocket, camera_id)

@app.get("/events")
def get_events(limit: int = 100):
    events = aws_service.get_events(limit)
    # Enrich with presigned URLs
    for e in events:
        if e.get("s3_key"):
            e["image_url"] = aws_service.generate_presigned_url(e["s3_key"])
    return events

@app.post("/events")
def create_event(event: Dict):
    """Receive event from camera client."""
    if aws_service.save_event(event):
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Failed to save event")

@app.delete("/events/{event_id}")
def delete_event(event_id: str):
    """Delete a single event."""
    if aws_service.delete_event(event_id):
        return {"status": "deleted", "event_id": event_id}
    raise HTTPException(status_code=500, detail="Failed to delete event")

class BulkDeleteRequest(BaseModel):
    event_ids: List[str]

@app.post("/events/delete-bulk")
def delete_events_bulk(request: BulkDeleteRequest):
    """Delete multiple events at once."""
    print(f"[DELETE-BULK] Received {len(request.event_ids)} event_ids to delete")
    if request.event_ids:
        print(f"[DELETE-BULK] First ID: {request.event_ids[0]}")
    result = aws_service.delete_events(request.event_ids)
    print(f"[DELETE-BULK] Result: {result}")
    return result

class EventUpdate(BaseModel):
    reviewed: Optional[bool] = None

@app.patch("/events/{event_id}")
def update_event(event_id: str, update: EventUpdate):
    """Update an event's fields (e.g., mark as reviewed)."""
    updates = update.dict(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    if aws_service.update_event(event_id, updates):
        return {"status": "updated", "event_id": event_id, "updates": updates}
    raise HTTPException(status_code=500, detail="Failed to update event")

class BulkReviewRequest(BaseModel):
    event_ids: List[str]

@app.post("/events/review-bulk")
def review_events_bulk(request: BulkReviewRequest):
    """Mark multiple events as reviewed at once."""
    reviewed_count = 0
    for event_id in request.event_ids:
        if aws_service.update_event(event_id, {"reviewed": True}):
            reviewed_count += 1
    return {"reviewed": reviewed_count, "total": len(request.event_ids)}

@app.get("/scan-network")
def scan_network():
    devices = scanner_service.scan_network()
    return {"devices": devices}

# ==================== FACE RECOGNITION ====================

class FaceRegister(BaseModel):
    name: str
    images_base64: List[str]  # List of Base64 encoded images

class FaceAddImages(BaseModel):
    images_base64: List[str]  # List of Base64 encoded images to add

class FaceSearch(BaseModel):
    image_base64: str  # Base64 encoded image

@app.get("/faces")
def list_faces():
    """List all registered persons."""
    persons = aws_service.list_faces()
    return persons

@app.post("/faces")
def register_face(face_data: FaceRegister):
    """Register a new person with one or more face images."""
    import base64
    
    if not face_data.images_base64:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    person_id = None
    results = []
    
    for idx, img_b64 in enumerate(face_data.images_base64):
        try:
            image_bytes = base64.b64decode(img_b64)
        except Exception:
            results.append({"index": idx, "success": False, "error": "Invalid base64 image"})
            continue
        
        result = aws_service.index_face(image_bytes, face_data.name, person_id)
        if result.get("success"):
            person_id = result.get("person_id")  # Use same person_id for subsequent images
            results.append({"index": idx, "success": True, "face_id": result.get("face_id")})
        else:
            results.append({"index": idx, "success": False, "error": result.get("error")})
    
    successful = [r for r in results if r.get("success")]
    if not successful:
        raise HTTPException(status_code=400, detail="No faces could be registered")
    
    return {
        "success": True,
        "person_id": person_id,
        "name": face_data.name,
        "faces_registered": len(successful),
        "results": results
    }

@app.post("/faces/{person_id}/images")
def add_face_images(person_id: str, face_data: FaceAddImages):
    """Add more images to an existing person."""
    import base64
    
    # Check if person exists
    person = aws_service._get_person_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    results = []
    for idx, img_b64 in enumerate(face_data.images_base64):
        try:
            image_bytes = base64.b64decode(img_b64)
        except Exception:
            results.append({"index": idx, "success": False, "error": "Invalid base64 image"})
            continue
        
        result = aws_service.index_face(image_bytes, person.get("name", ""), person_id)
        if result.get("success"):
            results.append({"index": idx, "success": True, "face_id": result.get("face_id")})
        else:
            results.append({"index": idx, "success": False, "error": result.get("error")})
    
    successful = [r for r in results if r.get("success")]
    return {
        "success": len(successful) > 0,
        "person_id": person_id,
        "faces_added": len(successful),
        "results": results
    }

@app.delete("/faces/{face_id}")
def delete_face(face_id: str):
    """Delete a single face image (keeps person if other images exist)."""
    if aws_service.delete_face(face_id):
        return {"status": "deleted", "face_id": face_id}
    raise HTTPException(status_code=500, detail="Failed to delete face")

@app.delete("/persons/{person_id}")
def delete_person(person_id: str):
    """Delete a person and all their face images."""
    if aws_service.delete_person(person_id):
        return {"status": "deleted", "person_id": person_id}
    raise HTTPException(status_code=500, detail="Failed to delete person")

@app.post("/faces/search")
def search_face(search_data: FaceSearch):
    """Search for a face in the registered collection."""
    import base64
    try:
        image_bytes = base64.b64decode(search_data.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    
    result = aws_service.search_faces(image_bytes)
    return result

# ==================== CAMERA STATUS ====================


def check_camera_connection(url: str, username: str = "", password: str = "") -> bool:
    """Check if a camera URL is reachable."""
    try:
        full_url = url
        if username and password:
            protocol, address = url.split("://", 1)
            full_url = f"{protocol}://{username}:{password}@{address}"
        
        # Try to open the stream
        cap = cv2.VideoCapture(full_url)
        if not cap.isOpened():
            # Try with /video appended
            if not url.endswith("/video"):
                full_url_video = f"{full_url}/video" if not full_url.endswith("/") else f"{full_url}video"
                cap = cv2.VideoCapture(full_url_video)
        
        is_open = cap.isOpened()
        if is_open:
            ret, _ = cap.read()
            cap.release()
            return ret
        cap.release()
        return False
    except Exception as e:
        print(f"Error checking camera: {e}")
        return False

@app.get("/cameras/{cam_id}/check")
def check_camera_status(cam_id: str):
    """Check if a specific camera is online."""
    cameras = aws_service.get_cameras()
    cam = next((c for c in cameras if c.get("camera_id") == cam_id), None)
    if not cam:
        return {"status": "NOT_FOUND"}
    
    url = cam.get("url", "")
    if not url:
        return {"status": "NO_URL"}
    
    username = cam.get("username", "")
    password = cam.get("password", "")
    
    is_online = check_camera_connection(url, username, password)
    
    # Update status in storage
    new_status = "ONLINE" if is_online else "OFFLINE"
    aws_service.update_camera(cam_id, {"status": new_status, "last_heartbeat": time.time() if is_online else 0})
    
    return {"status": new_status, "camera_id": cam_id}

@app.get("/cameras/check-all")
def check_all_cameras():
    """Check status of all cameras."""
    cameras = aws_service.get_cameras()
    results = []
    for cam in cameras:
        cam_id = cam.get("camera_id")
        url = cam.get("url", "")
        if url:
            username = cam.get("username", "")
            password = cam.get("password", "")
            is_online = check_camera_connection(url, username, password)
            new_status = "ONLINE" if is_online else "OFFLINE"
            aws_service.update_camera(cam_id, {"status": new_status, "last_heartbeat": time.time() if is_online else 0})
            results.append({"camera_id": cam_id, "status": new_status})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
