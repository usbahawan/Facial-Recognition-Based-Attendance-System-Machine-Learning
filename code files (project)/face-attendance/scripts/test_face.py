import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("../models/face_recognition_model.h5")

# Load class names (labels)
class_names = np.load("../data/labels.npy")

# Load a test image
img = cv2.imread("../dataset/Usbah/Usbah_20251018233326584534.jpg")  # replace with any image path
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (160,160))
img = np.expand_dims(img, 0) / 255.0

# Predict
preds = model.predict(img)
print("Predicted:", class_names[np.argmax(preds)], "Confidence:", np.max(preds))
