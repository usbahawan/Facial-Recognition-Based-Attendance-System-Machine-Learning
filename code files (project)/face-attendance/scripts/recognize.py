import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import sqlite3
from datetime import datetime
import time

# ================================
# Load trained model
# ================================
model = tf.keras.models.load_model("../models/face_recognition_model.h5")
IMG_SIZE = 160

# Load and sort class names
data_dir = "../dataset"
class_names = sorted(os.listdir(data_dir))

# ================================
# Initialize SQLite database
# ================================
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT
)
''')
conn.commit()

# ================================
# Track last attendance time per person
# ================================
last_marked_time = {}

# ================================
# Function to mark attendance
# ================================
def mark_attendance(name, confidence):
    if confidence < 0.60:  # only confident predictions
        return

    current_time = time.time()
    date_today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    # Skip if within 30 seconds
    if name in last_marked_time and current_time - last_marked_time[name] < 30:
        return

    # Check if already marked today
    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date_today))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                       (name, date_today, time_now))
        conn.commit()
        print(f"✅ Attendance marked for {name} at {time_now}")
    else:
        print(f"⚠️ {name} already marked today — waiting 30s for next update")

    last_marked_time[name] = current_time

# ================================
# Initialize webcam
# ================================
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("🎥 Starting camera... Press 'q' to quit.")

# ================================
# Real-time face recognition loop
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    face_imgs = []
    coords = []

    # Preprocess faces for batch prediction
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = img_to_array(face)
        face = face / 255.0
        face_imgs.append(face)
        coords.append((x, y, w, h))

    if face_imgs:
        face_imgs_np = np.array(face_imgs)
        preds = model.predict(face_imgs_np, verbose=0)  # batch prediction

        for i, pred in enumerate(preds):
            confidence = float(np.max(pred))
            name = class_names[np.argmax(pred)]
            x, y, w, h = coords[i]

            # Label unknown if low confidence
            name_display = "Unknown" if confidence < 0.60 else f"{name} ({confidence:.2f})"

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name_display, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

            # Mark attendance
            mark_attendance(name, confidence)

    cv2.imshow("Face Recognition - Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================================
# Cleanup
# ================================
cap.release()
cv2.destroyAllWindows()
conn.close()
print("🛑 Program ended and database saved.")
