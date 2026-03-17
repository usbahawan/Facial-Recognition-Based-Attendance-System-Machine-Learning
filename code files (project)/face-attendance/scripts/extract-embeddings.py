"""
Extract FaceNet embeddings from dataset/<label>/ folders
"""

import pathlib, cv2, numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import joblib
from sklearn.preprocessing import LabelEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
DATASET_DIR = pathlib.Path(__file__).parent.parent / "dataset"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load FaceNet models
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

embeddings = []
labels = []

label_names = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
print(f"[INFO] Found labels: {label_names}")

for label in label_names:
    folder = DATASET_DIR / label
    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".png"]:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)
        if face is not None:
            face = face.unsqueeze(0).to(DEVICE)
            emb = resnet(face).detach().cpu().numpy()
            embeddings.append(emb[0])
            labels.append(label)

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Save embeddings and label encoder
np.save(MODEL_DIR / "embeddings.npy", embeddings)
joblib.dump(labels_encoded, MODEL_DIR / "labels_encoded.joblib")
joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
print("[INFO] Embeddings and labels saved.")
