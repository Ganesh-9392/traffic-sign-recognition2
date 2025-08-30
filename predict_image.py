import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

def predict_image(image_path):
    model = load_model(MODEL_PATH)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Predicted Class: {class_id}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
    else:
        predict_image(sys.argv[1])
