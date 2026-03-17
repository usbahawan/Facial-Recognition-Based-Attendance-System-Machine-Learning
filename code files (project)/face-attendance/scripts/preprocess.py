"""
Preprocess dataset: read images from dataset/<label>/ folders,
limit to 100 per label, resize, normalize, shuffle, split into train/val,
and save as NumPy arrays.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pathlib

# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = pathlib.Path(__file__).parent.parent / "dataset"
OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Constants
# -----------------------------
IMG_SIZE = 160
MAX_IMAGES_PER_LABEL = 100

# -----------------------------
# Collect images
# -----------------------------
X, y = [], []

label_names = sorted([d for d in os.listdir(DATASET_DIR) if (DATASET_DIR / d).is_dir()])
print(f"[INFO] Found labels: {label_names}")

for label_idx, label_name in enumerate(label_names):
    folder = DATASET_DIR / label_name
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]

    # Limit images per label
    files = files[:MAX_IMAGES_PER_LABEL]

    for fname in files:
        fpath = folder / fname
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # color!
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_idx)

# -----------------------------
# Convert to arrays and normalize
# -----------------------------
X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

print(f"[INFO] Dataset shape: {X.shape}, labels: {y.shape}")

# -----------------------------
# Shuffle dataset
# -----------------------------
X, y = shuffle(X, y, random_state=42)

# -----------------------------
# Split into train/val (80/20)
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# Save preprocessed data
# -----------------------------
np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_val.npy", y_val)
np.save(OUTPUT_DIR / "labels.npy", np.array(label_names))

print(f"[INFO] Preprocessed data saved to {OUTPUT_DIR}/")
