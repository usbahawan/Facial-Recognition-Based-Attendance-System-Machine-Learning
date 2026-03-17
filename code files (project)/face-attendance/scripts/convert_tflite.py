# scripts/convert_tflite.py
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "face_recognition_model.h5"
TFLITE_PATH = Path(__file__).parent.parent / "models" / "face_recognition_model.tflite"
LABELS_PATH = Path(__file__).parent.parent / "data" / "labels.npy"  # saved earlier by preprocess.py

print("Loading Keras model:", MODEL_PATH)
model = tf.keras.models.load_model(str(MODEL_PATH))

# Option A: float32 tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
TFLITE_PATH.write_bytes(tflite_model)
print("Saved float32 TFLite model to:", TFLITE_PATH)

# Option B: optional post-training quantization (smaller & faster on CPU)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# For full integer quantization you would need a representative dataset function; using dynamic range quant here:
tflite_quant = converter.convert()
TFLITE_PATH.with_name("face_recognition_model_quant.tflite").write_bytes(tflite_quant)
print("Saved quantized TFLite model to:", TFLITE_PATH.with_name("face_recognition_model_quant.tflite"))

print("Done. Put the .tflite and labels file into your mobile app assets.")
