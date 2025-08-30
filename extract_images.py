# extract_images.py
import os, argparse, pickle
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = "data/GTSRB"
OUT_DIR = "test_images/gtsrb_samples"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20, help="How many images to export")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load training pickle (has many images)
    with open(os.path.join(DATA_DIR, "train.pickle"), "rb") as f:
        d = pickle.load(f)
    X = d["features"]
    y = d["labels"]

    # Load label names
    labels_df = pd.read_csv(os.path.join(DATA_DIR, "label_names.csv"))
    if "SignName" in labels_df.columns:
        name_col = "SignName"
    elif "Name" in labels_df.columns:
        name_col = "Name"
    else:
        raise ValueError("Couldn't find label name column in label_names.csv")
    id2name = dict(zip(labels_df["ClassId"], labels_df[name_col]))

    # Pick random indices
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(X), size=min(args.count, len(X)), replace=False)

    # Save images
    saved = []
    for i, idx in enumerate(idxs):
        img = X[idx].astype(np.uint8)         # RGB
        lab = int(y[idx])
        name = id2name.get(lab, str(lab)).replace("/", "-").replace(" ", "_")
        fname = f"sample_{i:03d}_class{lab}_{name}.jpg"
        path = os.path.join(OUT_DIR, fname)
        Image.fromarray(img).save(path, format="JPEG")
        saved.append((path, lab, name))

    print(f"Saved {len(saved)} images to {OUT_DIR}")
    for path, lab, name in saved:
        print(f"{path} -> {lab} ({name})")

if __name__ == "__main__":
    main()
