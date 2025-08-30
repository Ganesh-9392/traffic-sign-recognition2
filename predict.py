# predict.py
import numpy as np
import cv2
import sys
from tensorflow.keras.models import load_model
from src.config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH
import pickle, os

# Load label names (so we can print human-readable class)
label_path = "data/GTSRB/label_names.csv"

def load_labels():
    import pandas as pd
    if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        return dict(zip(df["ClassId"], df["SignName"]))   # ✅ correct
    else:
        return {i: str(i) for i in range(43)}


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 32, 32, 3)
    return img

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading labels...")
    labels = load_labels()

    print(f"Predicting for {image_path}...")
    img = preprocess_image(image_path)
    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    print(f"Predicted class: {class_id} ({labels.get(class_id, 'Unknown')}) with confidence {confidence:.2f}")

if __name__ == "__main__":
    main()
