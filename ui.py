from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from app import analyze_frame


st.set_page_config(
    page_title="AI Traffic Signal Optimization",
    page_icon="🚦",
    layout="wide",
)


@st.cache_resource
def load_model(model_name: str) -> YOLO:
    return YOLO(model_name)


def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def render_header() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(244, 196, 48, 0.25), transparent 30%),
                radial-gradient(circle at top right, rgba(19, 78, 94, 0.20), transparent 28%),
                linear-gradient(135deg, #f5efe2 0%, #eef3ea 55%, #dbe9ec 100%);
        }
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 24px;
            background: rgba(255, 252, 245, 0.82);
            border: 1px solid rgba(19, 78, 94, 0.12);
            box-shadow: 0 18px 60px rgba(19, 78, 94, 0.10);
            margin-bottom: 1rem;
        }
        .hero h1 {
            color: #133b45;
            margin-bottom: 0.2rem;
            font-size: 2.2rem;
        }
        .hero p {
            color: #335a63;
            font-size: 1rem;
            margin-bottom: 0;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(19, 78, 94, 0.10);
            text-align: center;
        }
        </style>
        <div class="hero">
            <h1>AI Traffic Signal Optimization</h1>
            <p>Upload a road image, detect vehicles with YOLOv8, and generate an adaptive green-signal recommendation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(vehicle_count: int, traffic_level: str, green_time: int) -> None:
    cols = st.columns(3)
    values = [
        ("Vehicles", str(vehicle_count)),
        ("Traffic Level", traffic_level),
        ("Green Time", f"{green_time} sec"),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem;color:#5c7075;">{label}</div>
                    <div style="font-size:1.8rem;color:#133b45;font-weight:700;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    render_header()

    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
        uploaded_file = st.file_uploader(
            "Upload traffic image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        run_clicked = st.button("Analyze Traffic", use_container_width=True)
        st.caption("The first run may download pretrained weights automatically.")

    if not uploaded_file:
        st.info("Upload a traffic image from the sidebar to start the demo.")
        return

    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Original Image", use_container_width=True)

    if not run_clicked:
        st.stop()

    model = load_model(model_name)
    temp_path = save_upload_to_temp(uploaded_file)
    frame = cv2.imread(str(temp_path))
    if frame is None:
        st.error("Could not read the uploaded image.")
        return

    with st.spinner("Detecting vehicles and optimizing signal timing..."):
        annotated, decision = analyze_frame(model, frame)

    render_metrics(
        decision.vehicle_count,
        decision.traffic_level,
        decision.green_time_seconds,
    )

    st.markdown("### Annotated Output")
    st.image(bgr_to_rgb(np.asarray(annotated)), use_container_width=True)

    st.markdown("### Signal Logic")
    st.write(
        f"For the detected density, the system recommends a green light duration of "
        f"`{decision.green_time_seconds}` seconds."
    )


if __name__ == "__main__":
    main()
