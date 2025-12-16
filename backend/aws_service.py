import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import base64

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DDB_TABLE = os.getenv("DDB_TABLE", "deteccion_eventos")
S3_BUCKET = os.getenv("S3_BUCKET", "valdivia-deteccion-prototipo")
FACE_COLLECTION_ID = os.getenv("FACE_COLLECTION_ID", "vision360-faces")
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "80"))

class AWSService:
    def __init__(self):
        self.use_local = False
        self.local_cameras = {}
        self.local_events = []
        self.local_faces = {}  # For local fallback
        try:
            self.session = boto3.session.Session(region_name=REGION)
            self.ddb = self.session.resource("dynamodb").Table(DDB_TABLE)
            self.s3 = self.session.client("s3")
            self.rekognition = self.session.client("rekognition")
            # Test connection
            self.ddb.scan(Limit=1)
            # Ensure face collection exists
            self._ensure_collection()
        except Exception as e:
            print(f"AWS DynamoDB unavailable: {e}")
            print("Using local in-memory storage as fallback.")
            self.use_local = True
            self.ddb = None
            self.s3 = None
            self.rekognition = None

    def _ensure_collection(self):
        """Create Rekognition collection if it doesn't exist."""
        try:
            self.rekognition.create_collection(CollectionId=FACE_COLLECTION_ID)
            print(f"Created face collection: {FACE_COLLECTION_ID}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                print(f"Face collection already exists: {FACE_COLLECTION_ID}")
            else:
                print(f"Error creating collection: {e}")

    # ==================== FACE RECOGNITION ====================
    
    def index_face(self, image_bytes: bytes, person_name: str, person_id: str = None) -> dict:
        """Add a face to the collection and store person metadata.
        
        If person_id is provided, add the face to an existing person.
        Otherwise, create a new person.
        """
        import uuid
        
        if self.use_local:
            face_id = str(uuid.uuid4())
            pid = person_id or str(uuid.uuid4())
            if pid not in self.local_faces:
                self.local_faces[pid] = {"person_id": pid, "name": person_name, "face_ids": []}
            self.local_faces[pid]["face_ids"].append(face_id)
            return {"success": True, "face_id": face_id, "person_id": pid, "name": person_name}
        
        try:
            response = self.rekognition.index_faces(
                CollectionId=FACE_COLLECTION_ID,
                Image={"Bytes": image_bytes},
                MaxFaces=1,
                QualityFilter="AUTO",
                DetectionAttributes=["DEFAULT"]
            )
            
            if not response.get("FaceRecords"):
                return {"success": False, "error": "No face detected in image"}
            
            face_id = response["FaceRecords"][0]["Face"]["FaceId"]
            pid = person_id or str(uuid.uuid4())
            
            # Check if person already exists
            existing = self._get_person_by_id(pid)
            if existing:
                # Add face_id to existing person's list
                face_ids = existing.get("face_ids", [])
                if isinstance(face_ids, str):
                    face_ids = [face_ids]
                face_ids.append(face_id)
                self.ddb.update_item(
                    Key={"event_id": f"PERSON#{pid}"},
                    UpdateExpression="SET face_ids = :fids",
                    ExpressionAttributeValues={":fids": face_ids}
                )
            else:
                # Create new person entry
                self.ddb.put_item(Item={
                    "event_id": f"PERSON#{pid}",
                    "timestamp": 0,
                    "person_id": pid,
                    "face_ids": [face_id],
                    "name": person_name
                })
            
            # Also store face-to-person mapping for quick lookups
            self.ddb.put_item(Item={
                "event_id": f"FACE#{face_id}",
                "timestamp": 0,
                "face_id": face_id,
                "person_id": pid,
                "name": person_name
            })
            
            return {"success": True, "face_id": face_id, "person_id": pid, "name": person_name}
        except Exception as e:
            print(f"Error indexing face: {e}")
            return {"success": False, "error": str(e)}

    def search_faces(self, image_bytes: bytes) -> dict:
        """Search for a face in the collection."""
        if self.use_local:
            return {"success": True, "match": False, "message": "Local mode - no face matching"}
        
        try:
            response = self.rekognition.search_faces_by_image(
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
                person = self._get_person_by_face_id(face_id)
                
                return {
                    "success": True,
                    "match": True,
                    "face_id": face_id,
                    "name": person.get("name", "Unknown"),
                    "similarity": similarity
                }
            else:
                return {"success": True, "match": False}
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidParameterException':
                return {"success": True, "match": False, "message": "No face detected"}
            print(f"Error searching faces: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"Error searching faces: {e}")
            return {"success": False, "error": str(e)}

    def _get_person_by_face_id(self, face_id: str) -> dict:
        """Get person metadata by face_id."""
        if self.use_local:
            for pid, person in self.local_faces.items():
                if face_id in person.get("face_ids", []):
                    return person
            return {}
        try:
            response = self.ddb.get_item(Key={"event_id": f"FACE#{face_id}"})
            return response.get("Item", {})
        except Exception as e:
            print(f"Error getting person: {e}")
            return {}
    
    def _get_person_by_id(self, person_id: str) -> dict:
        """Get person metadata by person_id."""
        if self.use_local:
            return self.local_faces.get(person_id, {})
        try:
            response = self.ddb.get_item(Key={"event_id": f"PERSON#{person_id}"})
            return response.get("Item", {})
        except Exception as e:
            print(f"Error getting person by id: {e}")
            return {}

    def list_faces(self) -> list:
        """List all registered persons with their face metadata."""
        if self.use_local:
            return list(self.local_faces.values())
        
        try:
            response = self.ddb.scan()
            # Return PERSON entries (grouped faces) instead of individual FACE entries
            persons = [i for i in response.get("Items", []) if i["event_id"].startswith("PERSON#")]
            return persons
        except Exception as e:
            print(f"Error listing faces: {e}")
            return []

    def delete_face(self, face_id: str) -> bool:
        """Delete a single face from collection (keeps person if other faces exist)."""
        if self.use_local:
            for pid, person in list(self.local_faces.items()):
                if face_id in person.get("face_ids", []):
                    person["face_ids"].remove(face_id)
                    if not person["face_ids"]:
                        del self.local_faces[pid]
                    return True
            return True
        
        try:
            # Delete from Rekognition collection
            self.rekognition.delete_faces(
                CollectionId=FACE_COLLECTION_ID,
                FaceIds=[face_id]
            )
            
            # Get person_id from face mapping
            face_meta = self._get_person_by_face_id(face_id)
            person_id = face_meta.get("person_id")
            
            # Delete face mapping
            self.ddb.delete_item(Key={"event_id": f"FACE#{face_id}"})
            
            # Update person's face_ids list
            if person_id:
                person = self._get_person_by_id(person_id)
                if person:
                    face_ids = person.get("face_ids", [])
                    if isinstance(face_ids, str):
                        face_ids = [face_ids]
                    if face_id in face_ids:
                        face_ids.remove(face_id)
                    if face_ids:
                        # Update with remaining faces
                        self.ddb.update_item(
                            Key={"event_id": f"PERSON#{person_id}"},
                            UpdateExpression="SET face_ids = :fids",
                            ExpressionAttributeValues={":fids": face_ids}
                        )
                    else:
                        # No faces left, delete person
                        self.ddb.delete_item(Key={"event_id": f"PERSON#{person_id}"})
            return True
        except Exception as e:
            print(f"Error deleting face: {e}")
            return False
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a person and all their faces."""
        if self.use_local:
            self.local_faces.pop(person_id, None)
            return True
        
        try:
            person = self._get_person_by_id(person_id)
            if not person:
                return False
            
            face_ids = person.get("face_ids", [])
            if isinstance(face_ids, str):
                face_ids = [face_ids]
            
            # Delete all faces from Rekognition
            if face_ids:
                self.rekognition.delete_faces(
                    CollectionId=FACE_COLLECTION_ID,
                    FaceIds=face_ids
                )
                # Delete face mappings
                for fid in face_ids:
                    self.ddb.delete_item(Key={"event_id": f"FACE#{fid}"})
            
            # Delete person entry
            self.ddb.delete_item(Key={"event_id": f"PERSON#{person_id}"})
            return True
        except Exception as e:
            print(f"Error deleting person: {e}")
            return False

    # ==================== CAMERAS ====================

    def get_cameras(self):
        if self.use_local:
            return list(self.local_cameras.values())
        try:
            resp = self.ddb.scan()
            return [i for i in resp.get("Items", []) if i["event_id"].startswith("CONFIG")]
        except Exception as e:
            print(f"Error fetching cameras: {e}")
            return list(self.local_cameras.values())

    def save_camera(self, cam_data):
        cam_id = cam_data.get("camera_id")
        if self.use_local:
            self.local_cameras[cam_id] = cam_data
            return True
        try:
            self.ddb.put_item(Item=cam_data)
            return True
        except Exception as e:
            print(f"Error saving camera: {e}")
            self.local_cameras[cam_id] = cam_data
            return True

    def delete_camera(self, cam_id):
        if self.use_local:
            self.local_cameras.pop(cam_id, None)
            return True
        try:
            self.ddb.delete_item(Key={"event_id": f"CONFIG#{cam_id}"})
            return True
        except Exception as e:
            print(f"Error deleting camera: {e}")
            self.local_cameras.pop(cam_id, None)
            return True

    def update_camera(self, cam_id, updates):
        if self.use_local:
            if cam_id in self.local_cameras:
                self.local_cameras[cam_id].update(updates)
                return True
            return False
        try:
            # Build UpdateExpression
            expr = "SET "
            vals = {}
            names = {}
            for k, v in updates.items():
                if k in ["camera_id", "event_id", "timestamp"]: continue
                key_placeholder = f"#{k}"
                val_placeholder = f":{k}"
                expr += f"{key_placeholder} = {val_placeholder}, "
                vals[val_placeholder] = v
                names[key_placeholder] = k
            
            expr = expr.rstrip(", ")
            
            self.ddb.update_item(
                Key={"event_id": f"CONFIG#{cam_id}"},
                UpdateExpression=expr,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=vals
            )
            return True
        except Exception as e:
            print(f"Error updating camera: {e}")
            # Fallback to local
            if cam_id in self.local_cameras:
                self.local_cameras[cam_id].update(updates)
                return True
            return False

    # ==================== EVENTS ====================

    def get_events(self, limit=100):
        if self.use_local:
            return sorted(self.local_events, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]
        try:
            resp = self.ddb.scan(Limit=limit)
            events = [i for i in resp.get("Items", []) if not i["event_id"].startswith("CONFIG") and not i["event_id"].startswith("FACE")]
            return sorted(events, key=lambda x: x.get("timestamp", 0), reverse=True)
        except Exception as e:
            print(f"Error fetching events: {e}")
            return sorted(self.local_events, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]

    def save_event(self, event_data):
        if self.use_local:
            self.local_events.append(event_data)
            # Keep only last 100 events in memory
            if len(self.local_events) > 100:
                self.local_events.pop(0)
            return True
        try:
            self.ddb.put_item(Item=event_data)
            return True
        except Exception as e:
            print(f"Error saving event: {e}")
            self.local_events.append(event_data)
            return True

    def delete_event(self, event_id: str) -> bool:
        """Delete a single event."""
        if self.use_local:
            self.local_events = [e for e in self.local_events if e.get("event_id") != event_id]
            return True
        try:
            self.ddb.delete_item(Key={"event_id": event_id})
            return True
        except Exception as e:
            print(f"Error deleting event: {e}")
            return False

    def update_event(self, event_id: str, updates: dict) -> bool:
        """Update an event's fields (e.g., reviewed status)."""
        if not updates:
            return False
            
        if self.use_local:
            for event in self.local_events:
                if event.get("event_id") == event_id:
                    event.update(updates)
                    return True
            return False
        
        try:
            # Build UpdateExpression
            expr = "SET "
            vals = {}
            names = {}
            for k, v in updates.items():
                if k in ["event_id", "timestamp"]: continue
                key_placeholder = f"#{k}"
                val_placeholder = f":{k}"
                expr += f"{key_placeholder} = {val_placeholder}, "
                vals[val_placeholder] = v
                names[key_placeholder] = k
            
            if not vals:
                return False
                
            expr = expr.rstrip(", ")
            
            self.ddb.update_item(
                Key={"event_id": event_id},
                UpdateExpression=expr,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=vals
            )
            return True
        except Exception as e:
            print(f"Error updating event: {e}")
            return False

    def delete_events(self, event_ids: list) -> dict:
        """Delete multiple events at once using batch operations."""
        if not event_ids:
            return {"deleted": 0, "failed": 0}
            
        print(f"[AWS] delete_events called with {len(event_ids)} IDs")
        
        if self.use_local:
            original_count = len(self.local_events)
            self.local_events = [e for e in self.local_events if e.get("event_id") not in event_ids]
            deleted = original_count - len(self.local_events)
            return {"deleted": deleted, "failed": 0}
        
        # Use batch_writer for faster bulk deletes (up to 25 items at a time)
        deleted = 0
        failed = 0
        try:
            with self.ddb.batch_writer() as batch:
                for event_id in event_ids:
                    batch.delete_item(Key={"event_id": event_id})
            # If we get here, all deletes were queued successfully
            deleted = len(event_ids)
            print(f"[AWS] Batch delete completed for {deleted} items")
        except Exception as e:
            print(f"[AWS] Batch delete error: {e}")
            # Fallback to individual deletes if batch fails
            for event_id in event_ids:
                try:
                    self.ddb.delete_item(Key={"event_id": event_id})
                    deleted += 1
                except Exception as inner_e:
                    print(f"[AWS] Individual delete failed for {event_id}: {inner_e}")
                    failed += 1
        
        return {"deleted": deleted, "failed": failed}


    def generate_presigned_url(self, key, expiration=3600):
        try:
            return self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': key},
                ExpiresIn=expiration
            )
        except ClientError as e:
            print(f"Error generating URL: {e}")
            return None

aws_service = AWSService()

