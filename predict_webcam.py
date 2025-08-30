import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import time
import os

# -----------------------------
# Load model & labels
# -----------------------------
MODEL_PATH = "models/tsr_model.keras"
LABELS_PATH = "data/GTSRB/label_names.csv"

print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading labels...")
df = pd.read_csv(LABELS_PATH)
labels = dict(zip(df["ClassId"], df["SignName"]))

# -----------------------------
# Preprocess function
# -----------------------------
def preprocess_image(img, target_size=(32, 32)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# Webcam setup
# -----------------------------
cap = cv2.VideoCapture(0)  # Try 1 or 2 if 0 doesn't work

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam started. Press 's' to save snapshot, 'q' to quit.")

# Create folder for snapshots
os.makedirs("snapshots", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Preprocess & predict
    input_img = preprocess_image(frame)
    preds = model.predict(input_img, verbose=0)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))
    label = labels[class_id]

    # Overlay prediction on frame
    text = f"{label} ({confidence:.2%})"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show window
    cv2.imshow("Traffic Sign Recognition", frame)

    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        # Save snapshot
        filename = f"snapshots/{label}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Saved snapshot: {filename}")

cap.release()
cv2.destroyAllWindows()
