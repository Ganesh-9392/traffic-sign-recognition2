# test_loader.py
from src.data import load_data

def main():
    X_train, X_val, y_train, y_val = load_data()
    print("SUCCESS: shapes:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)

if __name__ == "__main__":
    main()
