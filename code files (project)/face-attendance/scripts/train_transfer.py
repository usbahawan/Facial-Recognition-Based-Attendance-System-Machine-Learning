"""
Train face recognition model using transfer learning (MobileNetV2)
with partial layer unfreezing, augmentation, early stopping, and model checkpoint.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pathlib

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load preprocessed data
# -----------------------------
X_train = np.load(DATA_DIR / "X_train.npy")
X_val   = np.load(DATA_DIR / "X_val.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val   = np.load(DATA_DIR / "y_val.npy")
label_names = np.load(DATA_DIR / "labels.npy")

num_classes = len(label_names)
IMG_SIZE = 160
print(f"[INFO] Training on {num_classes} classes: {label_names}")

# -----------------------------
# Data augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7,1.3]
)

val_datagen = ImageDataGenerator()  # no augmentation for validation

# -----------------------------
# Build model (MobileNetV2)
# -----------------------------
base_model = MobileNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')

# Unfreeze last 20 layers to adapt to our small dataset
base_model.trainable = True
for layer in base_model.layers[:-20]:   # freeze first layers, train last 20
    layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
preds = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = ModelCheckpoint(MODEL_DIR / "face_recognition_model_best.h5",
                             monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# -----------------------------
# Training
# -----------------------------
EPOCHS = 25  # longer training
BATCH = 8

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH, shuffle=True),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# Save final model
# -----------------------------
model.save(MODEL_DIR / "face_recognition_model.h5")
print("[INFO] Model saved to /models/face_recognition_model.h5")
