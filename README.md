# Traffic Sign Recognition

Deep learning CNN model for classifying 43 traffic sign classes (GTSRB).
- Model: TensorFlow / Keras
- Web app: Streamlit (app.py) for image upload + top-5 predictions chart
- Real-time demo: predict_webcam.py using OpenCV

## Run locally
1. Create virtualenv and activate
2. Install requirements:
   pip install -r requirements.txt
3. Start web app:
   streamlit run app.py

> Note: dataset files are large and are not included in this repo. Place them under `data/GTSRB/` locally if needed.
