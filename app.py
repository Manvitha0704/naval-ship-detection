import streamlit as st
from PIL import Image
import os
import subprocess
import uuid
import pandas as pd

# -------------------------------------
# Page Configuration
# -------------------------------------
st.set_page_config(page_title="Naval Ship Detection", layout="centered")

# -------------------------------------
# Custom Styling
# -------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f2f8ff;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #003366;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .stButton > button {
        background-color: #003366;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------
# Title Section
# -------------------------------------
st.markdown('<div class="title">🚢 Naval Ship Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An Internship Project using YOLOv5 and ShipRSImageNet.v39i</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------
# Performance Metrics Display
# -------------------------------------
def load_performance_metrics():
    results_path = "runs/train/shiprs_yolov5s2/results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df.columns = [col.strip() for col in df.columns]
        latest = df.iloc[-1]

        try:
            metrics = {
                "Epoch": int(latest["epoch"]),
                "Precision (P)": round(latest["metrics/precision"], 3),
                "Recall (R)": round(latest["metrics/recall"], 3),
                "mAP@0.5": round(latest["metrics/mAP_0.5"], 3),
                "mAP@0.5:0.95": round(latest["metrics/mAP_0.5:0.95"], 3)
            }
            return metrics
        except KeyError as e:
            st.error(f"⚠️ Missing column in results.csv: {e}")
            return None
    else:
        return None

st.subheader("📈 Model Performance Summary")
metrics = load_performance_metrics()
if metrics:
    for key, value in metrics.items():
        st.markdown(f"**{key}:** {value}")
else:
    st.warning("Performance metrics not found or incomplete.")

st.markdown("---")

# -------------------------------------
# File Uploader
# -------------------------------------
st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader("Upload a .jpg/.png image of a ship", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save image temporarily
    image = Image.open(uploaded_file)
    image_id = str(uuid.uuid4())
    input_filename = f"input_{image_id}.jpg"
    image.save(input_filename)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Detection"):
        st.info("Running YOLOv5 detection... Please wait...")

        output_dir = f"runs/detect/streamlit_{image_id}"
        weights_path = "runs/train/shiprs_yolov5s2/weights/best.pt"

        command = [
            "python", "detect.py",
            "--weights", weights_path,
            "--source", input_filename,
            "--conf", "0.25",
            "--name", f"streamlit_{image_id}",
            "--save-txt"
        ]

        # Run YOLOv5 detection
        subprocess.run(command)

        output_filename = os.path.basename(input_filename)
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            st.success("✅ Detection complete!")
            st.image(output_path, caption="Detected Output", use_container_width=True)
        else:
            st.error("❌ Detection failed. Check if model or file path is correct.")

        os.remove(input_filename)

# -------------------------------------
# Footer
# -------------------------------------
st.markdown("""
<div class="footer">
🚀 Developed by Manvitha | Internship Project 2025
</div>
""", unsafe_allow_html=True)
