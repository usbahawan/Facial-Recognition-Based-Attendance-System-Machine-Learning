"""
Collect face images for a student in color.
Usage:
    python collect_images.py --name "Ali" --count 100
Press 'c' to capture current face crop. Press 'q' to quit.
"""

import cv2, os, argparse, random, numpy as np
from datetime import datetime

# ----------------------------
# Parse arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Student name (no spaces)")
parser.add_argument("--count", type=int, default=100, help="Number of images to capture (max 100)")
parser.add_argument("--output", default="../dataset", help="Output dataset directory")
args = parser.parse_args()

# ----------------------------
# Setup face detector
# ----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# Create label directory
# ----------------------------
label_dir = os.path.join(os.path.dirname(__file__), args.output, args.name)
os.makedirs(label_dir, exist_ok=True)

# ----------------------------
# Start webcam
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera. Make sure it's free and accessible.")

count = len([f for f in os.listdir(label_dir) if f.endswith(".jpg") or f.endswith(".png")])
print(f"Starting camera. Already {count} images. Target: {args.count}")

while count < args.count:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to capture", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    else:
        cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.putText(frame, f"{args.name} - Captured: {count}/{args.count}", 
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

    cv2.imshow("Collect Faces", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('c') and len(faces) > 0:
        # Crop face
        face_crop = frame[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Optional augmentation
        if random.random() < 0.5:
            face_crop = cv2.flip(face_crop, 1)  # horizontal flip
        factor = 0.8 + random.random()*0.4
        face_crop = np.clip(face_crop * factor, 0, 255).astype(np.uint8)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = os.path.join(label_dir, f"{args.name}_{timestamp}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
        count += 1
        print(f"[+] Saved {filename} ({count}/{args.count})")

print(f"[+] Target of {args.count} images reached or camera stopped. Exiting.")
cap.release()
cv2.destroyAllWindows()
