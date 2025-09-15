import streamlit as st
import numpy as np
import pandas as pd
import cv2
import altair as alt
from tensorflow.keras.models import load_model

# -----------------------------
# Load model and labels
# -----------------------------
@st.cache_resource
def load_trained_model():
    return load_model("models/tsr_model.keras")

@st.cache_data
def load_labels():
    df = pd.read_csv("data/GTSRB/label_names.csv")
    return dict(zip(df["ClassId"], df["SignName"]))

# -----------------------------
# Preprocess image
# -----------------------------
def preprocess_image(img, target_size=(32, 32)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="🚦 Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# -----------------------------
# Header Section
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 20px;">
        <h1 style="color:#FF4B4B;">🚦 Traffic Sign Recognition</h1>
        <p style="font-size:18px;">Upload a traffic sign image and let our deep learning model predict its class with confidence.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# File Uploader
# -----------------------------
uploaded_files = st.file_uploader(
    "📤 Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -----------------------------
# Prediction Section
# -----------------------------
model = tf.keras.models.load_model('models/traffic_model.h5')

with open('data/label_names.csv', 'r') as f:
    labels = f.readlines()


    for uploaded_file in uploaded_files:
        # Read file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Preprocess & predict
        input_img = preprocess_image(img)
        preds = model.predict(input_img, verbose=0)
        class_id = np.argmax(preds)
        confidence = float(np.max(preds))

        # Layout: Image on left, Prediction on right
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Uploaded Image", width=250)

        with col2:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#2ECC71; color:white; font-size:20px;">
                    ✅ Prediction: <b>{labels[class_id]}</b> (Class {class_id})<br>
                    Confidence: {confidence:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ✅ Probability Chart (Top 5 only with custom colors)
            prob_df = pd.DataFrame({
                "Class": [labels[i] for i in range(len(preds[0]))],
                "Probability": preds[0]
            }).sort_values("Probability", ascending=False).head(5)

            prob_df["Color"] = ["Top Prediction"] + ["Others"] * (len(prob_df) - 1)

            color_scale = alt.Scale(
                domain=["Top Prediction", "Others"],
                range=["#2ECC71", "#3498DB"]  # Green for top, Blue for others
            )

            chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadius=6)
                .encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Class:N", sort="-x"),
                    color=alt.Color("Color:N", scale=color_scale, legend=None),
                    tooltip=["Class", alt.Tooltip("Probability:Q", format=".2%")]
                )
            )

            text = chart.mark_text(
                align="left",
                baseline="middle",
                dx=3,
                color="white"
            ).encode(
                text=alt.Text("Probability:Q", format=".2%")
            )

            st.altair_chart(chart + text, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:14px; color:gray;'>Built with ❤️ using Streamlit & TensorFlow</p>",
        unsafe_allow_html=True
    )
