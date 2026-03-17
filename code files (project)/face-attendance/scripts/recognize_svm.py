"""
Live face recognition with attendance logging (robust)
"""

import torch, pathlib, cv2, numpy as np, joblib
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
from datetime import datetime

# -----------------------------
# Device and Models
# -----------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
svm = joblib.load(MODEL_DIR / "svm_model.joblib")
le = joblib.load(MODEL_DIR / "label_encoder.joblib")

# -----------------------------
# Database
# -----------------------------
DB_PATH = pathlib.Path(__file__).parent / "attendance.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        time TEXT
    )
""")
conn.commit()

# -----------------------------
# Function to log attendance
# -----------------------------
def log_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check if already logged today
    c.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if c.fetchone() is None:
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
        conn.commit()
        print(f"[INFO] {name} marked present at {time}")
    else:
        print(f"[INFO] {name} already logged today.")

# -----------------------------
# Webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)

    if face is not None:
        face = face.unsqueeze(0).to(DEVICE)
        emb = resnet(face).detach().cpu().numpy()
        probs = svm.predict_proba(emb)[0]
        idx = np.argmax(probs)
        name = le.inverse_transform([idx])[0]
        confidence = probs[idx]

        if confidence > 0.6:
            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            log_attendance(name)
        else:
            cv2.putText(frame, "Unknown", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
