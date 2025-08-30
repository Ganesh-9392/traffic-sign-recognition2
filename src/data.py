# src/data.py
import os
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from .config import DATA_DIR, NUM_CLASSES

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def load_data():
    print("Loading data from pickles...")

    train_path = os.path.join(DATA_DIR, "train.pickle")
    valid_path = os.path.join(DATA_DIR, "valid.pickle")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Missing file: {valid_path}")

    # Load train pickle (dict)
    train_data = load_pickle(train_path)
    X_train = np.array(train_data["features"])
    y_train = np.array(train_data["labels"])

    # Load validation pickle (dict)
    valid_data = load_pickle(valid_path)
    X_val = np.array(valid_data["features"])
    y_val = np.array(valid_data["labels"])

    # Normalize images to 0-1
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)

    print("Dataset loaded successfully!")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)

    return X_train, X_val, y_train, y_val
