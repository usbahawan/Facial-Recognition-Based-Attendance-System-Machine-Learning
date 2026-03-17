"""
Collect and crop face images from a folder of existing images.
Usage:
    python collect_images_from_folder.py --name "Ali" --input "../my_images/Ali" --output "../dataset"
"""

import cv2, os, argparse, numpy as np

# ----------------------------
# Parse arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Student name (no spaces)")
parser.add_argument("--input", required=True, help="Folder containing pre-existing images")
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
# Process each image
# ----------------------------
count = 0
for fname in os.listdir(args.input):
    fpath = os.path.join(args.input, fname)
    if not fname.lower().endswith((".jpg", ".png")):
        continue
    
    img = cv2.imread(fpath)
    if img is None:
        print(f"[!] Could not read {fpath}, skipping.")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    
    if len(faces) == 0:
        print(f"[!] No face detected in {fname}, skipping.")
        continue
    
    # Take the largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (160,160))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # Save cropped face
    out_fname = f"{args.name}_{count+1}.jpg"
    cv2.imwrite(os.path.join(label_dir, out_fname), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
    count += 1
    print(f"[+] Saved {out_fname}")

print(f"[INFO] Finished processing {count} face(s) for {args.name}.")
