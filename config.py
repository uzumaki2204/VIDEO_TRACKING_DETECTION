"""
Configuration hub for YOLO Vision Studio.
All paths, model configs, UI settings, and constants are defined here.
"""

from pathlib import Path
import os
import sys

# ─── Paths ───────────────────────────────────────────────────────────────────
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ASSETS_DIR = ROOT / "assets"
IMAGES_DIR = ROOT / "images"
VIDEOS_DIR = ROOT / "videos"
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"
TRACKING_RUNS_DIR = OUTPUTS_DIR / "tracking_runs"

# Ensure weights directory exists
WEIGHTS_DIR.mkdir(exist_ok=True)

# ─── App Metadata ────────────────────────────────────────────────────────────
APP_TITLE = "YOLO Vision Studio"
APP_ICON = "🔬"
APP_VERSION = "2.1.0"
APP_DESCRIPTION = (
    "Real-time Object Detection, Segmentation, Pose Estimation & Tracking "
    "powered by YOLO26, YOLO World v2, RT-DETR & Streamlit"
)

# ─── Inference Modes ─────────────────────────────────────────────────────────
MODE_IMAGE = "📷 Image Inference"
MODE_VIDEO = "🎬 Video Inference"
MODES_LIST = [MODE_IMAGE, MODE_VIDEO]

# ─── Tasks ───────────────────────────────────────────────────────────────────
TASK_DETECT = "Detection"
TASK_SEGMENT = "Segmentation"
TASK_WORLD = "YOLO World v2 (Text Prompt)"
TASK_YOLOE = "YOLOE (Text → Segmentation)"
TASK_POSE = "Pose Estimation"
TASKS_LIST = [TASK_DETECT, TASK_SEGMENT, TASK_WORLD, TASK_YOLOE, TASK_POSE]

# ─── Video Sources ───────────────────────────────────────────────────────────
SOURCE_STORED = "Stored Video"
SOURCE_WEBCAM = "Webcam"
SOURCE_RTSP = "RTSP Stream"
SOURCE_YOUTUBE = "YouTube"
VIDEO_SOURCES = [SOURCE_STORED, SOURCE_WEBCAM, SOURCE_RTSP, SOURCE_YOUTUBE]
STORED_MODE_SINGLE = "Single Video"
STORED_MODE_MULTI = "Multi Simultaneous"
STORED_MODE_QUEUE = "Queue Sequential"
STORED_VIDEO_MODES = [STORED_MODE_SINGLE, STORED_MODE_MULTI, STORED_MODE_QUEUE]

# ─── Compute Device ──────────────────────────────────────────────────────────
DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA0 = "cuda:0"
DEVICE_OPTIONS = {
    "Auto (prefer GPU)": DEVICE_AUTO,
    "CPU": DEVICE_CPU,
    "GPU (cuda:0)": DEVICE_CUDA0,
}
DEFAULT_DEVICE = DEVICE_AUTO

# ─── Model Catalog ────────────────────────────────────────────────────────────
# Each task has a dict of {display_label: model_filename}.
# Ultralytics auto-downloads any model not already in weights/.

DETECTION_MODELS = {
    "YOLO26-nano (fastest)": "yolo26n.pt",
    "YOLO26-small": "yolo26s.pt",
    "YOLO26-medium": "yolo26m.pt",
    "YOLO26-large": "yolo26l.pt",
    "YOLO26-xlarge (best accuracy)": "yolo26x.pt",
    "RT-DETR-Large (transformer)": "rtdetr-l.pt",
    "RT-DETR-XLarge (transformer)": "rtdetr-x.pt",
}

SEGMENTATION_MODELS = {
    "YOLO26-nano-seg (fastest)": "yolo26n-seg.pt",
    "YOLO26-small-seg": "yolo26s-seg.pt",
    "YOLO26-medium-seg": "yolo26m-seg.pt",
    "YOLO26-large-seg": "yolo26l-seg.pt",
    "YOLO26-xlarge-seg (best accuracy)": "yolo26x-seg.pt",
}

POSE_MODELS = {
    "YOLO26-nano-pose (fastest)": "yolo26n-pose.pt",
    "YOLO26-small-pose": "yolo26s-pose.pt",
    "YOLO26-medium-pose": "yolo26m-pose.pt",
    "YOLO26-large-pose": "yolo26l-pose.pt",
    "YOLO26-xlarge-pose (best accuracy)": "yolo26x-pose.pt",
}

WORLD_MODELS = {
    "YOLOv8-small-worldv2": "yolov8s-worldv2.pt",
    "YOLOv8-medium-worldv2": "yolov8m-worldv2.pt",
    "YOLOv8-large-worldv2 (recommended)": "yolov8l-worldv2.pt",
    "YOLOv8-xlarge-worldv2 (best accuracy)": "yolov8x-worldv2.pt",
}

YOLOE_MODELS = {
    "YOLOE-26n-seg (fastest)": "yoloe-26n-seg.pt",
    "YOLOE-26s-seg": "yoloe-26s-seg.pt",
    "YOLOE-26m-seg": "yoloe-26m-seg.pt",
    "YOLOE-26l-seg (recommended)": "yoloe-26l-seg.pt",
    "YOLOE-26x-seg (best accuracy)": "yoloe-26x-seg.pt",
}

# Defaults (first key in each dict)
DETECTION_MODEL = "yolo26n.pt"
SEGMENTATION_MODEL = "yolo26n-seg.pt"
POSE_MODEL = "yolo26n-pose.pt"

# YOLO World v2: open-vocabulary detection via natural language text prompts
YOLO_WORLD_MODEL = "yolov8l-worldv2.pt"

# YOLOE: open-vocabulary text-prompted detection + segmentation
YOLOE_MODEL = "yoloe-26l-seg.pt"

# ─── Default Assets ──────────────────────────────────────────────────────────
DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "office_4_detected.jpg"

# ─── Video Catalog ───────────────────────────────────────────────────────────
_VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.webm")
_VIDEO_SUFFIXES = {Path(pattern).suffix.lower() for pattern in _VIDEO_EXTENSIONS}


def resolve_video_path(path_value: str | Path | None = None) -> Path:
    """Resolve a video directory/file path from user input.

    Empty input falls back to the default ``videos/`` directory.
    Relative paths are resolved from the project root.
    """
    if path_value is None:
        return VIDEOS_DIR

    raw = str(path_value).strip()
    if not raw:
        return VIDEOS_DIR

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _video_label(path: Path, used_labels: set[str]) -> str:
    """Return a stable unique label for one discovered video file."""
    label = path.stem
    if label not in used_labels:
        used_labels.add(label)
        return label

    label = path.name
    if label not in used_labels:
        used_labels.add(label)
        return label

    suffix = 2
    while True:
        candidate = f"{path.stem} ({suffix})"
        if candidate not in used_labels:
            used_labels.add(candidate)
            return candidate
        suffix += 1


def get_videos_dict(path_value: str | Path | None = None) -> dict[str, Path]:
    """Scan a video directory or use a single video file if provided."""
    target = resolve_video_path(path_value)
    if not target.exists():
        return {}

    vids: dict[str, Path] = {}
    used_labels: set[str] = set()

    if target.is_file():
        if target.suffix.lower() not in _VIDEO_SUFFIXES:
            return {}
        vids[_video_label(target, used_labels)] = target
        return vids

    for ext in _VIDEO_EXTENSIONS:
        for p in sorted(target.glob(ext)):
            vids[_video_label(p, used_labels)] = p
    return dict(sorted(vids.items()))


# Kept for backward compat — but prefer get_videos_dict()
VIDEOS_DICT = get_videos_dict()

# ─── Inference Defaults ──────────────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.50
MIN_CONFIDENCE = 10  # slider min (%)
MAX_CONFIDENCE = 100  # slider max (%)
VIDEO_DISPLAY_WIDTH = 720
WEBCAM_PATH = 0
MIN_INFERENCE_IMGSZ = 512
MAX_INFERENCE_IMGSZ = 1536
INFERENCE_IMGSZ_STEP = 32
DEFAULT_INFERENCE_IMGSZ_CPU = 640
DEFAULT_INFERENCE_IMGSZ_GPU = 960
DEFAULT_GPU_HALF_PRECISION = True
MIN_UI_UPDATE_INTERVAL = 1
MAX_UI_UPDATE_INTERVAL = 8
DEFAULT_UI_UPDATE_INTERVAL_CPU = 1
DEFAULT_UI_UPDATE_INTERVAL_GPU = 4

# ─── Skip Frames ─────────────────────────────────────────────────────────────
DEFAULT_SKIP_FRAMES = 15  # process every frame
MIN_SKIP_FRAMES = 1
MAX_SKIP_FRAMES = 30

# ─── Tracking Crop Output ─────────────────────────────────────────────────────
DEFAULT_SAVE_TRACK_CROPS = True
TRACK_CROP_EXTENSION = ".jpg"

# ─── Tracker Config ──────────────────────────────────────────────────────────
TRACKER_BYTETRACK = "bytetrack.yaml"
TRACKER_BOTSORT = "botsort.yaml"
TRACKERS_LIST = [TRACKER_BYTETRACK, TRACKER_BOTSORT]
TRACK_GROUP_GAP_MS = 3000

# ─── Tracking Runtime ─────────────────────────────────────────────────────────
TRACK_RUNTIME_FAST = "Fast"
TRACK_RUNTIME_FULL = "Full"
TRACK_RUNTIME_MODES = [TRACK_RUNTIME_FAST, TRACK_RUNTIME_FULL]
DEFAULT_TRACK_RUNTIME_MODE = TRACK_RUNTIME_FAST
TRACK_LIVE_EXPORT_INTERVAL_FAST_S = 90.0
TRACK_LIVE_EXPORT_INTERVAL_FULL_S = 30.0
FAST_TRACKING_RETENTION_FRAMES = 1800

# ─── Video Processing Profiles ────────────────────────────────────────────────
VIDEO_PROFILE_INTERACTIVE = "Interactive"
VIDEO_PROFILE_BATCH_FAST = "Batch Fast (Long Video)"
VIDEO_PROCESSING_PROFILES = [
    VIDEO_PROFILE_INTERACTIVE,
    VIDEO_PROFILE_BATCH_FAST,
]
DEFAULT_VIDEO_PROCESSING_PROFILE = VIDEO_PROFILE_INTERACTIVE
BATCH_FAST_METRICS_UPDATE_INTERVAL = 240

# ─── YOLO World v2 Defaults ───────────────────────────────────────────────────
# Supports natural language prompts like "person in black", "red car", etc.
# DEFAULT_WORLD_CLASSES = "person, car, dog, cat, chair, table, laptop, phone"
DEFAULT_WORLD_CLASSES = "person"

# ─── YOLOE Defaults ──────────────────────────────────────────────────────────
# YOLOE supports category-level text prompts (NOT descriptive phrases).
# Unlike YOLO World v2, YOLOE provides detection + segmentation masks.
# DEFAULT_YOLOE_CLASSES = "person, car, dog, cat, chair, table, laptop, phone"
DEFAULT_YOLOE_CLASSES = "person"


def resolve_model_path(model_name: str) -> str:
    """Return local weights path if it exists, else the bare name for auto-download.

    After auto-download, call ``sweep_stray_weights()`` to move any
    ``.pt`` files that landed in the project root into ``weights/``.
    """
    local = WEIGHTS_DIR / model_name
    if local.exists():
        return str(local)
    # Not in weights/ yet — check project root (old download location)
    root_copy = ROOT / model_name
    if root_copy.exists():
        root_copy.rename(local)
        return str(local)
    # Will be auto-downloaded to CWD; return bare name
    return model_name


def sweep_stray_weights() -> None:
    """Move any ``.pt`` files from the project root into ``weights/``."""
    for pt_file in ROOT.glob("*.pt"):
        dest = WEIGHTS_DIR / pt_file.name
        if not dest.exists():
            pt_file.rename(dest)


def get_model_catalog(task: str) -> dict[str, str]:
    """Return ``{display_label: filename}`` for the given *task*."""
    _CATALOGS = {
        TASK_DETECT: DETECTION_MODELS,
        TASK_SEGMENT: SEGMENTATION_MODELS,
        TASK_POSE: POSE_MODELS,
        TASK_WORLD: WORLD_MODELS,
        TASK_YOLOE: YOLOE_MODELS,
    }
    return _CATALOGS.get(task, DETECTION_MODELS)


def get_device_catalog() -> dict[str, str]:
    """Return ``{display_label: device_value}`` for the device selector."""
    return DEVICE_OPTIONS.copy()
