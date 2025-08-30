# evaluate.py
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from src.config import MODEL_PATH, DATA_DIR, NUM_CLASSES

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_test_data():
    test_path = os.path.join(DATA_DIR, "test.pickle")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    test_data = load_pickle(test_path)
    X_test = np.array(test_data["features"])
    y_test = np.array(test_data["labels"])

    X_test = X_test.astype("float32") / 255.0
    y_test = to_categorical(y_test, NUM_CLASSES)

    return X_test, y_test

def main():
    print("Loading test data...")
    X_test, y_test = load_test_data()

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
