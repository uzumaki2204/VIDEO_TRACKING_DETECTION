"""
YOLO Vision Studio — Main Application
======================================
Real-time Object Detection, Segmentation, Pose Estimation & Tracking
powered by YOLO26, YOLO World v2 and Streamlit.
"""

import streamlit as st
import config
import model_loader
import image_service
import video_service

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption(f"v{config.APP_VERSION}")
    st.markdown("---")

    # 1️⃣  Inference mode ─────────────────────────────────────────────────────
    mode = st.radio("Inference Mode", config.MODES_LIST, key="mode")

    # 2️⃣  Task selection ─────────────────────────────────────────────────────
    task = st.radio("Task", config.TASKS_LIST, key="task")

    st.markdown("---")

    # 3️⃣  Model selector ─────────────────────────────────────────────────────
    catalog = config.get_model_catalog(task)
    model_label = st.selectbox(
        "🧠 Model",
        options=list(catalog.keys()),
        index=0,
        help="YOLO26 = fast CNN · RT-DETR = transformer (better on small objects) · World = open-vocab.",
        key="model_select",
    )
    selected_model = catalog[model_label]

    st.markdown("---")

    # 4️⃣  Compute device ─────────────────────────────────────────────────────
    device_catalog = config.get_device_catalog()
    device_label = st.selectbox(
        "Compute Device",
        options=list(device_catalog.keys()),
        index=0,
        help="Auto prefers CUDA when available. Explicit GPU selection falls back to CPU if CUDA is unavailable.",
        key="device_select",
    )
    selected_device = device_catalog[device_label]
    device_status = model_loader.get_device_status(selected_device)
    if device_status["using_gpu"]:
        st.caption(f"Runtime Device: GPU | {device_status['gpu_name']}")
    elif device_status["fallback"]:
        st.warning("CUDA was requested but is unavailable in this environment. Falling back to CPU.")
    else:
        st.caption("Runtime Device: CPU")

    st.markdown("---")

    # 5️⃣  Confidence slider ──────────────────────────────────────────────────
    confidence = (
        st.slider(
            "Model Confidence (%)",
            min_value=config.MIN_CONFIDENCE,
            max_value=config.MAX_CONFIDENCE,
            value=int(config.DEFAULT_CONFIDENCE * 100),
        )
        / 100.0
    )

# ── Main content ──────────────────────────────────────────────────────────────
st.title(config.APP_TITLE)
st.markdown(f"*{config.APP_DESCRIPTION}*")

if mode == config.MODE_IMAGE:
    image_service.render(task, confidence, selected_model, selected_device)
elif mode == config.MODE_VIDEO:
    video_service.render(task, confidence, selected_model, selected_device)
else:
    st.error("Please select a valid inference mode.")
