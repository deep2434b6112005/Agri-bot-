import cv2
import serial
import time
from ultralytics import YOLO

# LOAD MODEL
model = YOLO("runs/detect/train2/weights/best.pt")

# ARDUINO
arduino = serial.Serial("COM10", 9600, timeout=1)
time.sleep(2)

# ESP32 CAM STREAM
url = "http://192.168.4.1:81/stream"
cap = cv2.VideoCapture(url)

print("Press ENTER to capture | Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Stream failed")
        break

    cv2.imshow("ESP32 CAM", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # ENTER
        print("Capturing frame...")

        results = model(frame)[0]

        if len(results.boxes) > 0:
            confs = results.boxes.conf.tolist()
            best_index = confs.index(max(confs))
            cls_id = int(results.boxes.cls[best_index])
            disease = model.names[cls_id]
        else:
            disease = "Healthy"

        arduino.write((disease + "\n").encode())
        print("Detected:", disease)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
