import numpy as np
import os

# Go one level up from scripts/ to reach project root
dataset_path = "../dataset"
output_path = "../data/labels.npy"

labels = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])

np.save(output_path, np.array(labels))
print("✅ Labels saved to:", output_path)
print("Labels:", labels)
