import os
import numpy as np

# Paths
DATASET_DIR = "../dataset"  # adjust if needed
LABELS_FILE = "../data/labels.npy"
MIN_IMAGES_PER_LABEL = 30  # warning if less than this

# Load labels.npy
if not os.path.exists(LABELS_FILE):
    print(f"❌ Labels file not found: {LABELS_FILE}")
    labels = []
else:
    labels = np.load(LABELS_FILE)
    print(f"[INFO] Labels from labels.npy: {labels}")

# Check dataset folders
if not os.path.exists(DATASET_DIR):
    print(f"❌ Dataset folder not found: {DATASET_DIR}")
else:
    dataset_folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"[INFO] Folders in dataset/: {dataset_folders}")

    # Check each label
    for label in labels:
        folder_path = os.path.join(DATASET_DIR, label)
        if not os.path.exists(folder_path):
            print(f"❌ Folder for label '{label}' does NOT exist in dataset/")
        else:
            images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".png"))]
            print(f"[INFO] Label '{label}': {len(images)} images")
            if len(images) < MIN_IMAGES_PER_LABEL:
                print(f"⚠️  Warning: Less than {MIN_IMAGES_PER_LABEL} images for '{label}'")

    # Check for extra folders not in labels.npy
    for folder in dataset_folders:
        if folder not in labels:
            print(f"⚠️  Folder '{folder}' exists but not in labels.npy")
