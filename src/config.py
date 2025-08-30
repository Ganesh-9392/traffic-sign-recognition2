import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "GTSRB")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "tsr_model.keras")

# Image settings
IMG_HEIGHT = 30
IMG_WIDTH = 30
CHANNELS = 3
NUM_CLASSES = 43  # Number of traffic sign classes in GTSRB dataset

# Training settings
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
