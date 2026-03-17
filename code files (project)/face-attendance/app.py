# app.py
from flask import Flask, request, jsonify
import torch, pathlib, joblib, numpy as np, cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
from datetime import datetime
import io

app = Flask(__name__)

# -----------------------------
# Device and Models
# -----------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

MODEL_DIR = pathlib.Path(__file__).parent / "models"
svm = joblib.load(MODEL_DIR / "svm_model.joblib")
le = joblib.load(MODEL_DIR / "label_encoder.joblib")
labels = np.array([str(l) for l in le.classes_])

# -----------------------------
# Database
# -----------------------------
DB_PATH = pathlib.Path(__file__).parent / "scripts" / "attendance.db"

def log_attendance(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now()
    date = timestamp.strftime("%Y-%m-%d")
    time = timestamp.strftime("%H:%M:%S")

    # avoid duplicate attendance for same person same day
    c.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if c.fetchone() is None:
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
        conn.commit()
    conn.close()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# Recognition API
# -----------------------------
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    face = mtcnn(img)
    if face is None:
        return jsonify({'name': 'Unknown', 'confidence': 0.0})

    face = face.unsqueeze(0).to(DEVICE)
    emb = resnet(face).detach().cpu().numpy()
    probs = svm.predict_proba(emb)[0]
    idx = np.argmax(probs)
    confidence = float(probs[idx])
    name = labels[idx] if confidence > 0.6 else 'Unknown'

    if name != 'Unknown':
        timestamp = log_attendance(name)
    else:
        timestamp = None

    return jsonify({'name': name, 'confidence': confidence, 'timestamp': timestamp})

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
