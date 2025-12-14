import cv2
import time

url = "http://192.168.2.105:8081/video"

print(f"Testing connection to {url}...")
start = time.time()
cap = cv2.VideoCapture(url)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("SUCCESS: Frame read successfully")
    else:
        print("FAILURE: Opened but could not read frame")
else:
    print("FAILURE: Could not open stream")
cap.release()
print(f"Time taken: {time.time() - start:.2f}s")
