# extract_sample.py
import os, pickle, numpy as np, cv2, argparse

DATA_PICKLE = os.path.join("data", "GTSRB", "test.pickle")
OUT_DIR = "test_images"
os.makedirs(OUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--indices", nargs="*", type=int, default=[0,5,10,20],
                    help="list of indices to extract (e.g. --indices 0 5 10)")
parser.add_argument("--count", type=int, default=None,
                    help="save first N images (overrides --indices)")
args = parser.parse_args()

with open(DATA_PICKLE, "rb") as f:
    data = pickle.load(f)

X = np.array(data["features"])
y = np.array(data["labels"])

if args.count is not None:
    idxs = list(range(args.count))
else:
    idxs = args.indices

for idx in idxs:
    if idx < 0 or idx >= len(X):
        print("skip index out of range:", idx)
        continue
    img = X[idx]
    # convert to uint8 0-255 if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    # convert RGB->BGR for cv2
    try:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception:
        img_bgr = img
    out_path = os.path.join(OUT_DIR, f"sample_{idx}.jpg")
    ok = cv2.imwrite(out_path, img_bgr)
    print("Saved:", out_path, "label:", int(y[idx]), "ok:", ok)
