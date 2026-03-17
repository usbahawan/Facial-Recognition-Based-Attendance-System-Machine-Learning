from flask import Flask, render_template, request, jsonify
import sqlite3, base64, numpy as np, io, os
from PIL import Image
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

# -------------------------------------------------
# 1️⃣ Load Model and Labels
# -------------------------------------------------
MODEL_PATH = "../models/face_recognition_model.h5"
LABELS_PATH = "../data/labels.npy"
DB_PATH = "../scripts/attendance.db"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found at ../models/face_recognition_model.h5")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("❌ Labels not found at ../data/labels.npy")

MODEL = tf.keras.models.load_model(MODEL_PATH)
LABELS = list(np.load(LABELS_PATH, allow_pickle=True))
print("✅ Model and labels loaded successfully!")

# -------------------------------------------------
# 2️⃣ Attendance Data Retrieval
# -------------------------------------------------
def get_attendance():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, date, time FROM attendance ORDER BY date DESC, time DESC")
    data = cursor.fetchall()
    conn.close()
    return data

@app.route('/')
def index():
    records = get_attendance()
    return render_template('index.html', records=records)

# -------------------------------------------------
# 3️⃣ TensorFlow Prediction Function
# -------------------------------------------------
def predict_from_image_bytes(img_bytes):
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    pil = pil.resize((160, 160))
    arr = np.array(pil) / 255.0
    arr = np.expand_dims(arr, axis=0).astype('float32')

    preds = MODEL.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    name = LABELS[idx]
    return name, confidence

# -------------------------------------------------
# 4️⃣ Mobile Capture + Recognition
# -------------------------------------------------
@app.route("/mobile_capture", methods=["GET", "POST"])
def capture_from_mobile():
    if request.method == "GET":
        return render_template("mobile_capture.html")

    # POST method: process image
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "message": "No image received."}), 400

    try:
        image_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error decoding image: {str(e)}"}), 400

    # Predict using your TensorFlow model
    name, confidence = predict_from_image_bytes(img_bytes)

    if not name or confidence < 0.75:
        return jsonify({"success": False, "message": "Face not recognized clearly."}), 200

    # ✅ Mark attendance in SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, today))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
            (name, today, datetime.now().strftime("%H:%M:%S"))
        )
        conn.commit()
        msg = f"✅ Attendance marked for {name}"
    else:
        msg = f"⚠️ {name} already marked today"

    conn.close()

    return jsonify({"success": True, "name": name, "confidence": float(confidence), "message": msg})


# -------------------------------------------------
# 5️⃣ Run Flask App
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
