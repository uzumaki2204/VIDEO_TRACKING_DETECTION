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


def _inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 28%),
                radial-gradient(circle at top left, rgba(16, 185, 129, 0.08), transparent 24%),
                linear-gradient(180deg, #f5f7fb 0%, #eef3f8 100%);
        }
        .hero-panel {
            padding: 1.15rem 1.25rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 12px 36px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #0f766e;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.55;
        }
        .summary-card {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.92);
            padding: 0.9rem 1rem;
            min-height: 108px;
        }
        .summary-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .summary-value {
            font-size: 1.05rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .summary-note {
            font-size: 0.88rem;
            color: #475569;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_main_header(
    *,
    mode: str,
    task: str,
    model_label: str,
    device_label: str,
    confidence: float,
) -> None:
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">Vision Workflow</div>
            <div class="hero-title">{config.APP_ICON} {config.APP_TITLE}</div>
            <div class="hero-subtitle">{config.APP_DESCRIPTION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Mode", mode, "Chọn luồng xử lý ảnh tĩnh hoặc video."),
        ("Task", task, "Tác vụ suy luận đang được cấu hình."),
        ("Model", model_label, "Kiến trúc và checkpoint dùng để infer."),
        ("Runtime", device_label, f"Confidence {int(confidence * 100)}%."),
    ]
    for column, (label, value, note) in zip((col1, col2, col3, col4), cards):
        with column:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-label">{label}</div>
                    <div class="summary-value">{value}</div>
                    <div class="summary-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_ui_styles()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption(f"v{config.APP_VERSION}")
    st.caption("Thiết lập theo từng nhóm để giảm thao tác dò tìm trong sidebar.")

    with st.expander("1. Workflow", expanded=True):
        mode = st.radio("Inference Mode", config.MODES_LIST, key="mode")
        task = st.radio("Task", config.TASKS_LIST, key="task")

    with st.expander("2. Model & Runtime", expanded=True):
        catalog = config.get_model_catalog(task)
        model_label = st.selectbox(
            "Model",
            options=list(catalog.keys()),
            index=0,
            help="YOLO26 = fast CNN · RT-DETR = transformer · World / YOLOE = open-vocabulary.",
            key="model_select",
        )
        selected_model = catalog[model_label]

        device_catalog = config.get_device_catalog()
        device_label = st.selectbox(
            "Compute Device",
            options=list(device_catalog.keys()),
            index=0,
            help="Auto ưu tiên CUDA khi khả dụng; chọn GPU sẽ tự fallback CPU nếu môi trường không có CUDA.",
            key="device_select",
        )
        selected_device = device_catalog[device_label]
        device_status = model_loader.get_device_status(selected_device)
        if device_status["using_gpu"]:
            st.success(f"Runtime Device: GPU | {device_status['gpu_name']}")
        elif device_status["fallback"]:
            st.warning(
                "CUDA was requested but is unavailable in this environment. Falling back to CPU."
            )
        else:
            st.info("Runtime Device: CPU")

        confidence = (
            st.slider(
                "Model Confidence (%)",
                min_value=config.MIN_CONFIDENCE,
                max_value=config.MAX_CONFIDENCE,
                value=int(config.DEFAULT_CONFIDENCE * 100),
                help="Tăng để lọc bớt detection yếu, giảm để ưu tiên recall.",
            )
            / 100.0
        )

# ── Main content ──────────────────────────────────────────────────────────────
_render_main_header(
    mode=mode,
    task=task,
    model_label=model_label,
    device_label=device_label,
    confidence=confidence,
)

if mode == config.MODE_IMAGE:
    image_service.render(task, confidence, selected_model, selected_device)
elif mode == config.MODE_VIDEO:
    video_service.render(task, confidence, selected_model, selected_device)
else:
    st.error("Please select a valid inference mode.")
