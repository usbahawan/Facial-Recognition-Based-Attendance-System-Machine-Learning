"""
Train SVM classifier on FaceNet embeddings
"""

import numpy as np
from sklearn.svm import SVC
import joblib
import pathlib

MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"

# Load embeddings
embeddings = np.load(MODEL_DIR / "embeddings.npy")
labels_encoded = joblib.load(MODEL_DIR / "labels_encoded.joblib")

# Train SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(embeddings, labels_encoded)

# Save model
joblib.dump(svm, MODEL_DIR / "svm_model.joblib")
print("[INFO] SVM model trained and saved.")
