import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
else:
    print("✅ Cámara abierta correctamente. Presiona ESC para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame.")
            break
        cv2.imshow("Prueba de cámara", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
