"""
Video inference service — Stored videos, Webcam, RTSP & YouTube.

Features
--------
* Object detection, segmentation, YOLO World v2 & pose estimation
* ByteTrack / BoTSORT tracking with per-box track IDs (class | conf | ID:N)
* Local (per-frame) + Global (cumulative) tracking metrics
* Skip-frames slider for faster inferencing
* Multi-video simultaneous detection in side-by-side columns
* Browser-based webcam via streamlit-webrtc
"""

from __future__ import annotations

import base64
import csv
import html
import os
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yt_dlp

import config
from model_loader import get_model_for_task, load_fresh_model, resolve_device


# ── Track-ID colour palette (16 distinct BGR colours) ────────────────────────

_TRACK_COLORS = [
    (46, 204, 113),  # emerald
    (52, 152, 219),  # peter river
    (231, 76, 60),  # alizarin
    (241, 196, 15),  # sun flower
    (155, 89, 182),  # amethyst
    (26, 188, 156),  # turquoise
    (230, 126, 34),  # carrot
    (52, 73, 94),  # wet asphalt
    (22, 160, 133),  # green sea
    (39, 174, 96),  # nephritis
    (41, 128, 185),  # belize hole
    (142, 68, 173),  # wisteria
    (243, 156, 18),  # orange
    (211, 84, 0),  # pumpkin
    (192, 57, 43),  # pomegranate
    (127, 140, 141),  # asbestos
]
_TRACK_PREVIEW_MAX_SIDE = 320
_TRACK_PREVIEW_JPEG_QUALITY = 85
_PREVIEW_HIDDEN_COLUMNS = {"preview", "preview_crop_path"}
_TRACK_CROP_FILE_RE = re.compile(
    r"^(?P<image_id>.+)__track_(?P<track_id>\d+)__segment_(?P<segment_index>\d+)\.[^.]+$"
)


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Return a distinct BGR colour for *track_id*."""
    return _TRACK_COLORS[abs(track_id) % len(_TRACK_COLORS)]


@dataclass(frozen=True)
class _PerformanceOptions:
    imgsz: int
    half: bool
    ui_update_interval: int


@dataclass(frozen=True)
class _TrackObservation:
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class _TrackingEvent:
    video_name: str
    queue_index: int
    track_id: int
    segment_index: int
    class_name: str
    confidence: float
    frame_num: int
    image_id: str
    timestamp_ms: int
    timestamp_text: str


@dataclass(frozen=True)
class _TrackingSegment:
    video_name: str
    queue_index: int
    track_id: int
    segment_index: int
    segment_id: str
    class_name: str
    start_frame: int
    end_frame: int
    image_id_start: str
    image_id_end: str
    start_time_ms: int
    end_time_ms: int
    start_time_text: str
    end_time_text: str
    duration_ms: int
    duration_text: str
    hits: int
    representative_image_id: str | None
    representative_crop_path: str | None
    representative_preview_data_url: str | None


@dataclass(frozen=True)
class _CaptureTimingMeta:
    fps: float | None
    total_frames: int | None
    duration_ms: int | None


@dataclass(frozen=True)
class _VideoRunResult:
    video_name: str
    queue_index: int
    queue_total: int
    frames_read: int
    processed_frames: int
    skipped_frames: int
    raw_events: list[_TrackingEvent]
    grouped_segments: list[_TrackingSegment]
    unique_track_count: int
    output_dir: str | None = None
    saved_crop_count: int = 0
    error: str | None = None


@dataclass
class _TrackSegmentRuntime:
    segment_index: int
    last_seen_ms: int


@dataclass(frozen=True)
class _SavedCropInfo:
    image_id: str
    crop_path: str | None
    preview_data_url: str | None


@dataclass(frozen=True)
class _TrackingRuntimeOptions:
    mode: str
    keep_full_history: bool
    live_export_interval_s: float
    eager_live_snapshot_on_new_crop: bool
    recent_track_window_frames: int | None


@dataclass(frozen=True)
class _VideoProcessingProfileOptions:
    mode: str
    render_live_preview: bool
    show_source_preview: bool
    use_grab_skip: bool
    metrics_update_interval: int
    allow_tracking_exports: bool
    force_fast_tracking: bool


class _TrackingMetricsState:
    def __init__(
        self,
        *,
        retain_all_time: bool,
        recent_window_frames: int | None = None,
    ) -> None:
        self.retain_all_time = retain_all_time
        self.recent_window_frames = recent_window_frames
        self.tracked_ids: set[int] = set()
        self.class_tracked: dict[str, set[int]] = defaultdict(set)
        self._window_tracked_ids: set[int] = set()
        self._window_class_tracked: dict[str, set[int]] = defaultdict(set)
        self._last_seen_by_track: dict[int, tuple[str, int]] = {}

    @property
    def tracked_total(self) -> int:
        return len(self.tracked_ids)

    def observe(self, track_id: int, class_name: str, frame_num: int) -> None:
        self.tracked_ids.add(track_id)
        self.class_tracked.setdefault(class_name, set()).add(track_id)
        self._window_tracked_ids.add(track_id)
        self._window_class_tracked.setdefault(class_name, set()).add(track_id)
        if not self.retain_all_time:
            self._last_seen_by_track[track_id] = (class_name, frame_num)

    def prune(self, frame_num: int) -> None:
        if self.retain_all_time or self.recent_window_frames is None:
            return

        stale_before = frame_num - self.recent_window_frames
        stale_track_ids = [
            track_id
            for track_id, (_, last_seen_frame) in self._last_seen_by_track.items()
            if last_seen_frame < stale_before
        ]
        for track_id in stale_track_ids:
            class_name, _ = self._last_seen_by_track.pop(track_id)
            self._window_tracked_ids.discard(track_id)
            class_ids = self._window_class_tracked.get(class_name)
            if class_ids is None:
                continue
            class_ids.discard(track_id)
            if not class_ids:
                self._window_class_tracked.pop(class_name, None)


# ── Custom annotation with track IDs on bounding boxes ───────────────────────


def _annotate_with_ids(
    frame: np.ndarray,
    result,
    enable_tracking: bool,
    font_scale: float = 0.50,
    box_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes with ``class | conf% | ID:N`` labels.

    For segmentation / pose tasks, masks / keypoints are rendered first
    via ``result.plot(labels=False, boxes=False)`` then custom box
    labels with track IDs are overlaid.
    """
    has_masks = getattr(result, "masks", None) is not None and len(result.masks)
    has_kpts = getattr(result, "keypoints", None) is not None and len(result.keypoints)

    if has_masks or has_kpts:
        annotated = result.plot(labels=False, boxes=False, conf=False)
    else:
        annotated = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    names = result.names
    boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    track_ids = None
    if enable_tracking and result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (box, cls_id, conf) in enumerate(zip(boxes_xyxy, classes, confs)):
        x1, y1, x2, y2 = box
        tid = track_ids[i] if track_ids is not None else None
        color = _color_for_track(tid) if tid is not None else _color_for_track(cls_id)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Label: "class | 87% | ID:5"
        name = names[cls_id]
        parts = [name, f"{conf:.0%}"]
        if tid is not None:
            parts.append(f"ID:{tid}")
        label = " | ".join(parts)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - th - 4),
            (x1 + tw + 6, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ── Frame → JPEG bytes (avoids Streamlit MediaFileHandler cache issue) ───────


def _frame_to_bytes(frame: np.ndarray) -> bytes:
    """Encode a BGR *frame* to JPEG bytes for ``st.image()``.

    Sending raw bytes avoids Streamlit's internal temp-file caching,
    which can cause ``MediaFileStorageError`` during fast video loops.
    """
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _resize_for_display(frame: np.ndarray) -> np.ndarray:
    """Resize an annotated frame for UI display without shrinking inference size."""
    h_orig, w_orig = frame.shape[:2]
    target_w = config.VIDEO_DISPLAY_WIDTH
    if w_orig <= target_w:
        return frame
    target_h = int(target_w * h_orig / w_orig)
    return cv2.resize(frame, (target_w, target_h))


def _default_inference_imgsz(device: str) -> int:
    if device.startswith("cuda"):
        return config.DEFAULT_INFERENCE_IMGSZ_GPU
    return config.DEFAULT_INFERENCE_IMGSZ_CPU


def _default_ui_update_interval(device: str) -> int:
    if device.startswith("cuda"):
        return config.DEFAULT_UI_UPDATE_INTERVAL_GPU
    return config.DEFAULT_UI_UPDATE_INTERVAL_CPU


def _performance_options(device: str) -> _PerformanceOptions:
    """Collect runtime knobs that shift work from CPU-bound UI to GPU inference."""
    using_gpu = device.startswith("cuda")
    with st.sidebar.expander("4. Performance", expanded=using_gpu):
        imgsz = st.slider(
            "Inference Size",
            min_value=config.MIN_INFERENCE_IMGSZ,
            max_value=config.MAX_INFERENCE_IMGSZ,
            value=_default_inference_imgsz(device),
            step=config.INFERENCE_IMGSZ_STEP,
            help=(
                "Larger sizes send more work to the GPU and can improve small-object "
                "recall, but they use more VRAM and may reduce FPS."
            ),
            key="perf_imgsz",
        )
        half = st.checkbox(
            "FP16 Inference",
            value=using_gpu and config.DEFAULT_GPU_HALF_PRECISION,
            disabled=not using_gpu,
            help=(
                "Use half precision on CUDA. This usually improves throughput and "
                "reduces VRAM usage."
            ),
            key="perf_half",
        )
        ui_update_interval = st.slider(
            "UI Refresh Every N Processed Frames",
            min_value=config.MIN_UI_UPDATE_INTERVAL,
            max_value=config.MAX_UI_UPDATE_INTERVAL,
            value=_default_ui_update_interval(device),
            help=(
                "Inference still runs on every processed frame. This only reduces "
                "overlay drawing, JPEG encoding, and Streamlit refresh frequency so "
                "the GPU spends less time waiting on CPU work."
            ),
            key="perf_ui_update_interval",
        )

    return _PerformanceOptions(
        imgsz=imgsz,
        half=using_gpu and half,
        ui_update_interval=ui_update_interval,
    )


def _format_timestamp_ms(value_ms: int | float | None, show_millis: bool = True) -> str:
    """Format milliseconds as ``HH:MM:SS.mmm`` or ``HH:MM:SS``."""
    if value_ms is None or value_ms < 0:
        return "—"

    total_ms = int(round(value_ms))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    seconds, millis = divmod(rem_ms, 1000)
    if show_millis:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_elapsed_seconds(seconds: float) -> str:
    return _format_timestamp_ms(max(seconds, 0.0) * 1000, show_millis=False)


def _format_progress(progress: float | None) -> str:
    if progress is None:
        return "—"
    return f"{progress * 100:.1f}%"


def _capture_timing_meta(vid_cap: cv2.VideoCapture) -> _CaptureTimingMeta:
    """Read FPS / frame count metadata once for timeline calculations."""
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else None

    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = total_frames if total_frames > 0 else None

    duration_ms = None
    if fps and total_frames:
        duration_ms = int(round((total_frames / fps) * 1000))

    return _CaptureTimingMeta(fps=fps, total_frames=total_frames, duration_ms=duration_ms)


def _current_video_ms(
    vid_cap: cv2.VideoCapture,
    meta: _CaptureTimingMeta,
    frame_num: int,
) -> int | None:
    """Return the current video position in milliseconds."""
    pos_ms = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms and pos_ms > 0:
        return int(round(pos_ms))
    if meta.fps:
        return int(round((frame_num / meta.fps) * 1000))
    return None


def _skip_frames_by_grab(vid_cap: cv2.VideoCapture, frames_to_skip: int) -> int:
    """Advance a capture by grabbing frames without decoding them."""
    skipped = 0
    for _ in range(max(frames_to_skip, 0)):
        if not vid_cap.grab():
            break
        skipped += 1
    return skipped


def _video_progress(current_ms: int | None, meta: _CaptureTimingMeta) -> float | None:
    if current_ms is None or meta.duration_ms is None or meta.duration_ms <= 0:
        return None
    return min(max(current_ms / meta.duration_ms, 0.0), 1.0)


def _supports_half_precision(task: str) -> bool:
    """Return whether a task should use runtime FP16 inference."""
    return task != config.TASK_YOLOE


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "item"


def _segment_id(track_id: int, segment_index: int) -> str:
    return f"track_{track_id}_segment_{segment_index:03d}"


def _format_output_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def _create_tracking_run_dir(task: str) -> Path:
    config.TRACKING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_safe_path_component(task)}"
    run_dir = config.TRACKING_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _video_output_dir(run_output_dir: Path, video_name: str) -> Path:
    video_dir = run_output_dir / _safe_path_component(video_name)
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def _resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = config.ROOT / path
    return path


def _crop_box_from_frame(
    frame: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def _resize_preview_image(image: np.ndarray, max_side: int = _TRACK_PREVIEW_MAX_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    largest_side = max(h, w)
    if largest_side <= max_side:
        return image

    scale = max_side / float(largest_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _crop_to_data_url(crop: np.ndarray) -> str | None:
    preview = _resize_preview_image(crop)
    ok, encoded = cv2.imencode(
        ".jpg",
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), _TRACK_PREVIEW_JPEG_QUALITY],
    )
    if not ok:
        return None
    encoded_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_b64}"


def _data_url_to_bytes(data_url: str) -> bytes | None:
    if not data_url.startswith("data:") or ";base64," not in data_url:
        return None
    _, encoded = data_url.split(";base64,", maxsplit=1)
    try:
        return base64.b64decode(encoded)
    except Exception:
        return None


def _data_url_to_image_payload(data_url: str) -> tuple[str, bytes] | None:
    if not data_url.startswith("data:") or ";base64," not in data_url:
        return None
    mime_header, encoded = data_url.split(";base64,", maxsplit=1)
    mime_type = mime_header.removeprefix("data:") or "image/jpeg"
    try:
        return mime_type, base64.b64decode(encoded)
    except Exception:
        return None


def _mime_type_to_extension(mime_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime_type.lower(), ".bin")


def _mime_type_from_extension(extension: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(extension.lower(), "application/octet-stream")


def _crop_path_to_data_url(crop_path: str) -> str:
    crop_source = _resolve_project_path(crop_path)
    if crop_source is None or not crop_source.exists() or not crop_source.is_file():
        return ""
    encoded = base64.b64encode(crop_source.read_bytes()).decode("ascii")
    mime_type = _mime_type_from_extension(crop_source.suffix)
    return f"data:{mime_type};base64,{encoded}"


def _preview_export_filename(
    row: dict[str, object],
    row_index: int,
    extension: str,
    used_names: set[str],
) -> str:
    name_parts = [
        row.get("video_name"),
        row.get("segment_id"),
        row.get("timestamp") or row.get("start_time"),
    ]
    base_name = "__".join(
        _safe_path_component(str(part))
        for part in name_parts
        if part not in (None, "")
    )
    if not base_name:
        base_name = f"preview_{row_index:04d}"

    candidate = f"{base_name}{extension}"
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}_{suffix:02d}{extension}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _preview_display_columns(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []
    return [column for column in rows[0].keys() if column not in _PREVIEW_HIDDEN_COLUMNS]


def _preview_crop_path_from_row(row: dict[str, object]) -> str:
    return str(
        row.get("preview_crop_path")
        or row.get("representative_crop_path")
        or ""
    ).strip()


def _preview_export_document(
    title: str,
    row_count: int,
    table_headers: list[str],
    body_rows: list[str],
) -> str:
    header_html = "".join(
        (
            '<th draggable="true" title="Drag to reorder columns">'
            f"{html.escape(column.replace('_', ' ').title())}"
            "</th>"
        )
        for column in table_headers
    )
    table_html = (
        "<tbody>" + "".join(body_rows) + "</tbody>"
        if body_rows
        else '<tbody><tr><td colspan="99" class="empty">No rows available.</td></tr></tbody>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    body {{
      margin: 24px;
      background: #f6f8fb;
      color: #172033;
    }}
    h1 {{
      margin-bottom: 8px;
      font-size: 28px;
    }}
    p {{
      margin-top: 0;
      margin-bottom: 20px;
      color: #4b5565;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: #ffffff;
      border: 1px solid #d8e0ea;
      border-radius: 16px;
      box-shadow: 0 12px 30px rgba(23, 32, 51, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 960px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #e7edf5;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef3f9;
      font-weight: 700;
      white-space: nowrap;
      z-index: 1;
      cursor: grab;
      user-select: none;
    }}
    th.dragging {{
      opacity: 0.55;
      cursor: grabbing;
    }}
    th.drop-target {{
      background: #dbe9f8;
      box-shadow: inset 0 -3px 0 #2f6fed;
    }}
    tr:nth-child(even) td {{
      background: #fbfcfe;
    }}
    img {{
      display: block;
      max-width: 128px;
      max-height: 96px;
      border-radius: 10px;
      border: 1px solid #d8e0ea;
      object-fit: cover;
      background: #ffffff;
    }}
    a {{
      color: inherit;
      text-decoration: none;
    }}
    .empty {{
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Rows: {row_count}. Click any thumbnail to open the linked image file. Drag any column header to reorder the table.</p>
  <div class="table-wrap">
    <table>
      <thead><tr>{header_html}</tr></thead>
      {table_html}
    </table>
  </div>
  <script>
    (() => {{
      const table = document.querySelector("table");
      if (!table) return;

      const getHeaders = () => Array.from(table.querySelectorAll("thead th"));
      const getIndex = (cell) => Array.from(cell.parentElement.children).indexOf(cell);
      let dragHeader = null;

      const clearDragState = () => {{
        getHeaders().forEach((header) => {{
          header.classList.remove("dragging", "drop-target");
        }});
      }};

      const moveColumn = (fromIndex, toIndex) => {{
        if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0) return;
        table.querySelectorAll("tr").forEach((row) => {{
          const cells = Array.from(row.children);
          if (fromIndex >= cells.length || toIndex >= cells.length) return;

          const movingCell = cells[fromIndex];
          const targetCell = cells[toIndex];
          if (!movingCell || !targetCell || movingCell === targetCell) return;

          if (fromIndex < toIndex) {{
            row.insertBefore(movingCell, targetCell.nextSibling);
          }} else {{
            row.insertBefore(movingCell, targetCell);
          }}
        }});
      }};

      getHeaders().forEach((header) => {{
        header.addEventListener("dragstart", (event) => {{
          dragHeader = header;
          header.classList.add("dragging");
          if (event.dataTransfer) {{
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/plain", header.textContent || "");
          }}
        }});

        header.addEventListener("dragover", (event) => {{
          event.preventDefault();
          if (!dragHeader || dragHeader === header) return;
          header.classList.add("drop-target");
          if (event.dataTransfer) {{
            event.dataTransfer.dropEffect = "move";
          }}
        }});

        header.addEventListener("dragleave", () => {{
          header.classList.remove("drop-target");
        }});

        header.addEventListener("drop", (event) => {{
          event.preventDefault();
          if (!dragHeader || dragHeader === header) {{
            clearDragState();
            return;
          }}
          moveColumn(getIndex(dragHeader), getIndex(header));
          clearDragState();
          dragHeader = null;
        }});

        header.addEventListener("dragend", () => {{
          clearDragState();
          dragHeader = null;
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


def _build_preview_export_zip(
    title: str,
    rows: list[dict[str, object]],
    *,
    archive_root_name: str,
) -> bytes:
    preview_column = "preview"
    archive_root = _safe_path_component(archive_root_name)
    display_columns = _preview_display_columns(rows)
    used_image_names: set[str] = set()
    preview_path_lookup: dict[str, str] = {}
    body_rows: list[str] = []
    image_dir = f"{archive_root}/images"

    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row_index, row in enumerate(rows, start=1):
            crop_path = _preview_crop_path_from_row(row)
            preview_data_url = str(row.get(preview_column) or "").strip()
            preview_cell_html = '<span class="empty">—</span>'
            lookup_key = crop_path or preview_data_url
            if lookup_key:
                image_rel_path = preview_path_lookup.get(lookup_key)
                if image_rel_path is None:
                    image_bytes: bytes | None = None
                    extension = ".jpg"
                    crop_source = _resolve_project_path(crop_path)
                    if crop_source is not None and crop_source.exists() and crop_source.is_file():
                        image_bytes = crop_source.read_bytes()
                        extension = crop_source.suffix or extension
                    elif preview_data_url:
                        payload = _data_url_to_image_payload(preview_data_url)
                        if payload is not None:
                            mime_type, image_bytes = payload
                            extension = _mime_type_to_extension(mime_type)
                    if image_bytes is not None:
                        image_name = _preview_export_filename(
                            row,
                            row_index,
                            extension,
                            used_image_names,
                        )
                        image_zip_path = f"{image_dir}/{image_name}"
                        archive.writestr(image_zip_path, image_bytes)
                        image_rel_path = f"images/{image_name}"
                        preview_path_lookup[lookup_key] = image_rel_path
                if image_rel_path is not None:
                    escaped_path = html.escape(image_rel_path, quote=True)
                    preview_cell_html = (
                        f'<a href="{escaped_path}" target="_blank" rel="noopener">'
                        f'<img src="{escaped_path}" alt="Preview" loading="lazy"></a>'
                    )

            cells = [f"<td>{preview_cell_html}</td>"]
            for column in display_columns:
                value = row.get(column)
                text = "" if value is None else str(value)
                cells.append(f"<td>{html.escape(text)}</td>")
            body_rows.append("<tr>" + "".join(cells) + "</tr>")

        document = _preview_export_document(
            title,
            len(rows),
            ["Preview", *display_columns],
            body_rows,
        )
        archive.writestr(f"{archive_root}/index.html", document.encode("utf-8"))

    archive_buffer.seek(0)
    return archive_buffer.getvalue()


def _write_preview_export_directory(
    title: str,
    rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    preview_column = "preview"
    display_columns = _preview_display_columns(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    used_image_names: set[str] = set()
    preview_path_lookup: dict[str, str] = {}
    body_rows: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        crop_path = _preview_crop_path_from_row(row)
        preview_data_url = str(row.get(preview_column) or "").strip()
        preview_cell_html = '<span class="empty">—</span>'
        lookup_key = crop_path or preview_data_url

        if lookup_key:
            image_rel_path = preview_path_lookup.get(lookup_key)
            if image_rel_path is None:
                crop_source = _resolve_project_path(crop_path)
                if crop_source is not None and crop_source.exists() and crop_source.is_file():
                    image_rel_path = os.path.relpath(crop_source, output_dir).replace("\\", "/")
                    preview_path_lookup[lookup_key] = image_rel_path
                elif preview_data_url:
                    payload = _data_url_to_image_payload(preview_data_url)
                    if payload is not None:
                        mime_type, image_bytes = payload
                        images_dir.mkdir(parents=True, exist_ok=True)
                        image_name = _preview_export_filename(
                            row,
                            row_index,
                            _mime_type_to_extension(mime_type),
                            used_image_names,
                        )
                        image_path = images_dir / image_name
                        image_path.write_bytes(image_bytes)
                        image_rel_path = f"images/{image_name}"
                        preview_path_lookup[lookup_key] = image_rel_path
            if image_rel_path is not None:
                escaped_path = html.escape(image_rel_path, quote=True)
                preview_cell_html = (
                    f'<a href="{escaped_path}" target="_blank" rel="noopener">'
                    f'<img src="{escaped_path}" alt="Preview" loading="lazy"></a>'
                )

        cells = [f"<td>{preview_cell_html}</td>"]
        for column in display_columns:
            value = row.get(column)
            text = "" if value is None else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    document = _preview_export_document(
        title,
        len(rows),
        ["Preview", *display_columns],
        body_rows,
    )
    html_path = output_dir / "index.html"
    html_path.write_text(document, encoding="utf-8")
    return html_path


def _save_segment_crop(
    crop: np.ndarray,
    crops_dir: Path,
    image_id: str,
    track_id: int,
    segment_index: int,
) -> str | None:
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_name = (
        f"{_safe_path_component(image_id)}__track_{track_id}"
        f"__segment_{segment_index:03d}{config.TRACK_CROP_EXTENSION}"
    )
    crop_path = crops_dir / crop_name
    if not cv2.imwrite(str(crop_path), crop):
        return None
    return _format_output_path(crop_path)


def _image_id(video_name: str, frame_num: int) -> str:
    return f"{video_name}:{frame_num:06d}"


def _build_tracking_events(
    observations: list[_TrackObservation],
    video_name: str,
    queue_index: int,
    frame_num: int,
    timestamp_ms: int | None,
    frame: np.ndarray,
    segment_runtime: dict[tuple[str, int, int], _TrackSegmentRuntime],
    saved_crops: dict[tuple[str, int, int, int], _SavedCropInfo],
    save_track_crops: bool = False,
    crops_dir: Path | None = None,
) -> list[_TrackingEvent]:
    """Attach video/frame/timestamp metadata to raw tracking observations."""
    if not observations:
        return []

    safe_timestamp_ms = int(timestamp_ms or 0)
    timestamp_text = _format_timestamp_ms(safe_timestamp_ms)
    image_id = _image_id(video_name, frame_num)

    events: list[_TrackingEvent] = []
    for obs in observations:
        runtime_key = (video_name, queue_index, obs.track_id)
        state = segment_runtime.get(runtime_key)
        is_new_segment = (
            state is None
            or safe_timestamp_ms - state.last_seen_ms > config.TRACK_GROUP_GAP_MS
        )
        segment_index = 1 if state is None else state.segment_index
        if is_new_segment and state is not None:
            segment_index += 1

        segment_runtime[runtime_key] = _TrackSegmentRuntime(
            segment_index=segment_index,
            last_seen_ms=safe_timestamp_ms,
        )

        if is_new_segment:
            crop = _crop_box_from_frame(frame, obs.bbox_xyxy)
            if crop is not None:
                crop_path: str | None = None
                if save_track_crops and crops_dir is not None:
                    crop_path = _save_segment_crop(
                        crop,
                        crops_dir,
                        image_id=image_id,
                        track_id=obs.track_id,
                        segment_index=segment_index,
                    )
                preview_data_url = None if crop_path else _crop_to_data_url(crop)
                if preview_data_url or crop_path:
                    saved_crops[(video_name, queue_index, obs.track_id, segment_index)] = (
                        _SavedCropInfo(
                            image_id=image_id,
                            crop_path=crop_path,
                            preview_data_url=preview_data_url,
                        )
                    )

        events.append(
            _TrackingEvent(
                video_name=video_name,
                queue_index=queue_index,
                track_id=obs.track_id,
                segment_index=segment_index,
                class_name=obs.class_name,
                confidence=obs.confidence,
                frame_num=frame_num,
                image_id=image_id,
                timestamp_ms=safe_timestamp_ms,
                timestamp_text=timestamp_text,
            )
        )

    return events


def _finalize_segment(
    events: list[_TrackingEvent],
    crop_info: _SavedCropInfo | None = None,
) -> _TrackingSegment:
    """Create one grouped segment from consecutive raw tracking events."""
    start = events[0]
    end = events[-1]
    majority_class = Counter(event.class_name for event in events).most_common(1)[0][0]
    duration_ms = max(0, end.timestamp_ms - start.timestamp_ms)
    return _TrackingSegment(
        video_name=start.video_name,
        queue_index=start.queue_index,
        track_id=start.track_id,
        segment_index=start.segment_index,
        segment_id=_segment_id(start.track_id, start.segment_index),
        class_name=majority_class,
        start_frame=start.frame_num,
        end_frame=end.frame_num,
        image_id_start=start.image_id,
        image_id_end=end.image_id,
        start_time_ms=start.timestamp_ms,
        end_time_ms=end.timestamp_ms,
        start_time_text=start.timestamp_text,
        end_time_text=end.timestamp_text,
        duration_ms=duration_ms,
        duration_text=_format_timestamp_ms(duration_ms),
        hits=len(events),
        representative_image_id=crop_info.image_id if crop_info else start.image_id,
        representative_crop_path=crop_info.crop_path if crop_info else None,
        representative_preview_data_url=crop_info.preview_data_url if crop_info else None,
    )


def _group_tracking_events(
    events: list[_TrackingEvent],
    saved_crops: dict[tuple[str, int, int, int], _SavedCropInfo] | None = None,
) -> list[_TrackingSegment]:
    """Group raw events into continuous track segments using the configured gap."""
    grouped: list[_TrackingSegment] = []
    events_by_track: dict[tuple[str, int, int, int], list[_TrackingEvent]] = defaultdict(list)
    for event in events:
        events_by_track[
            (event.video_name, event.queue_index, event.track_id, event.segment_index)
        ].append(event)

    for key, track_events in events_by_track.items():
        ordered = sorted(track_events, key=lambda item: (item.timestamp_ms, item.frame_num))
        crop_info = saved_crops.get(key) if saved_crops else None
        grouped.append(_finalize_segment(ordered, crop_info=crop_info))

    return sorted(
        grouped,
        key=lambda item: (
            item.queue_index,
            item.start_time_ms,
            item.track_id,
            item.segment_index,
        ),
    )


def _result_export_rows(result: _VideoRunResult) -> list[dict[str, str | int | float]]:
    """Flatten one video run result into combined raw + grouped CSV rows."""
    rows: list[dict[str, str | int | float]] = []

    for event in result.raw_events:
        rows.append(
            {
                "row_type": "raw",
                "video_name": event.video_name,
                "queue_index": event.queue_index,
                "track_id": event.track_id,
                "segment_index": event.segment_index,
                "segment_id": _segment_id(event.track_id, event.segment_index),
                "class_name": event.class_name,
                "confidence": round(event.confidence, 6),
                "frame_num": event.frame_num,
                "image_id": event.image_id,
                "timestamp_ms": event.timestamp_ms,
                "timestamp_text": event.timestamp_text,
                "start_frame": "",
                "end_frame": "",
                "image_id_start": "",
                "image_id_end": "",
                "start_time_ms": "",
                "end_time_ms": "",
                "start_time_text": "",
                "end_time_text": "",
                "duration_ms": "",
                "duration_text": "",
                "hits": "",
                "representative_image_id": "",
                "representative_crop_path": "",
                "output_run_dir": result.output_dir or "",
            }
        )

    for segment in result.grouped_segments:
        rows.append(
            {
                "row_type": "grouped",
                "video_name": segment.video_name,
                "queue_index": segment.queue_index,
                "track_id": segment.track_id,
                "segment_index": segment.segment_index,
                "segment_id": segment.segment_id,
                "class_name": segment.class_name,
                "confidence": "",
                "frame_num": "",
                "image_id": "",
                "timestamp_ms": "",
                "timestamp_text": "",
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "image_id_start": segment.image_id_start,
                "image_id_end": segment.image_id_end,
                "start_time_ms": segment.start_time_ms,
                "end_time_ms": segment.end_time_ms,
                "start_time_text": segment.start_time_text,
                "end_time_text": segment.end_time_text,
                "duration_ms": segment.duration_ms,
                "duration_text": segment.duration_text,
                "hits": segment.hits,
                "representative_image_id": segment.representative_image_id or "",
                "representative_crop_path": segment.representative_crop_path or "",
                "output_run_dir": result.output_dir or "",
            }
        )

    return rows


def _results_export_rows(results: list[_VideoRunResult]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for result in results:
        rows.extend(_result_export_rows(result))
    return rows


def _rows_to_csv(rows: list[dict[str, str | int | float]]) -> str:
    """Serialize export rows to CSV text."""
    fieldnames = [
        "row_type",
        "video_name",
        "queue_index",
        "track_id",
        "segment_index",
        "segment_id",
        "class_name",
        "confidence",
        "frame_num",
        "image_id",
        "timestamp_ms",
        "timestamp_text",
        "start_frame",
        "end_frame",
        "image_id_start",
        "image_id_end",
        "start_time_ms",
        "end_time_ms",
        "start_time_text",
        "end_time_text",
        "duration_ms",
        "duration_text",
        "hits",
        "representative_image_id",
        "representative_crop_path",
        "output_run_dir",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _build_preview_rows(
    results: list[_VideoRunResult],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    successful = [result for result in results if result.error is None]
    preview_lookup = {
        (
            segment.video_name,
            segment.queue_index,
            segment.track_id,
            segment.segment_index,
        ): {
            "preview": segment.representative_preview_data_url or "",
            "crop_path": segment.representative_crop_path or "",
        }
        for result in successful
        for segment in result.grouped_segments
    }
    grouped_preview_rows = [
        {
            "preview": segment.representative_preview_data_url or "",
            "preview_crop_path": segment.representative_crop_path or "",
            "video_name": segment.video_name,
            "queue_index": segment.queue_index,
            "track_id": segment.track_id,
            "segment_id": segment.segment_id,
            "class_name": segment.class_name,
            "start_time": segment.start_time_text,
            "end_time": segment.end_time_text,
            "duration": segment.duration_text,
            "start_frame": segment.start_frame,
            "end_frame": segment.end_frame,
            "image_id_start": segment.image_id_start,
            "image_id_end": segment.image_id_end,
            "representative_image_id": segment.representative_image_id,
            "representative_crop_path": segment.representative_crop_path,
            "hits": segment.hits,
        }
        for result in successful
        for segment in result.grouped_segments
    ]
    raw_preview_rows = [
        {
            "preview": preview_lookup.get(
                (
                    event.video_name,
                    event.queue_index,
                    event.track_id,
                    event.segment_index,
                ),
                {},
            ).get("preview", ""),
            "preview_crop_path": preview_lookup.get(
                (
                    event.video_name,
                    event.queue_index,
                    event.track_id,
                    event.segment_index,
                ),
                {},
            ).get("crop_path", ""),
            "video_name": event.video_name,
            "queue_index": event.queue_index,
            "track_id": event.track_id,
            "segment_id": _segment_id(event.track_id, event.segment_index),
            "class_name": event.class_name,
            "confidence": event.confidence,
            "frame_num": event.frame_num,
            "image_id": event.image_id,
            "timestamp": event.timestamp_text,
        }
        for result in successful
        for event in result.raw_events
    ]
    return grouped_preview_rows, raw_preview_rows


def _safe_int(value: object, default: int = 0) -> int:
    try:
        text = "" if value is None else str(value).strip()
        return int(text) if text else default
    except (TypeError, ValueError):
        return default


def _build_track_gallery_rows(grouped_preview_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    gallery_by_track: dict[tuple[str, int, int], dict[str, object]] = {}

    for row in grouped_preview_rows:
        video_name = str(row.get("video_name") or "").strip()
        queue_index = _safe_int(row.get("queue_index"), 0)
        track_id = _safe_int(row.get("track_id"), -1)
        if track_id < 0:
            continue

        segment_id = str(row.get("segment_id") or "").strip()
        class_name = str(row.get("class_name") or "").strip() or "unknown"
        start_time = str(row.get("start_time") or "").strip()
        end_time = str(row.get("end_time") or "").strip()
        representative_image_id = str(row.get("representative_image_id") or "").strip()
        preview_data_url = str(row.get("preview") or "").strip()
        preview_crop_path = _preview_crop_path_from_row(row)
        hits = max(_safe_int(row.get("hits"), 1), 1)

        key = (video_name, queue_index, track_id)
        existing = gallery_by_track.get(key)
        if existing is None:
            gallery_by_track[key] = {
                "preview": preview_data_url,
                "preview_crop_path": preview_crop_path,
                "video_name": video_name,
                "queue_index": queue_index,
                "track_id": track_id,
                "class_name": class_name,
                "segments": 1,
                "total_hits": hits,
                "first_seen": start_time,
                "last_seen": end_time or start_time,
                "representative_segment_id": segment_id,
                "representative_image_id": representative_image_id,
            }
            continue

        existing["segments"] = int(existing["segments"]) + 1
        existing["total_hits"] = int(existing["total_hits"]) + hits
        if end_time:
            existing["last_seen"] = end_time
        if not str(existing.get("preview") or "").strip() and preview_data_url:
            existing["preview"] = preview_data_url
        if not str(existing.get("preview_crop_path") or "").strip() and preview_crop_path:
            existing["preview_crop_path"] = preview_crop_path
        if str(existing.get("class_name") or "").strip() in ("", "unknown") and class_name:
            existing["class_name"] = class_name
        if not str(existing.get("representative_segment_id") or "").strip() and segment_id:
            existing["representative_segment_id"] = segment_id
        if (
            not str(existing.get("representative_image_id") or "").strip()
            and representative_image_id
        ):
            existing["representative_image_id"] = representative_image_id

    return sorted(
        gallery_by_track.values(),
        key=lambda item: (
            _safe_int(item.get("queue_index"), 0),
            _safe_int(item.get("track_id"), 0),
            str(item.get("video_name") or ""),
        ),
    )


def _persist_video_tracking_outputs(
    result: _VideoRunResult,
    video_output_dir: Path,
    *,
    live: bool = False,
    write_archives: bool = True,
) -> None:
    if result.error is not None or (not result.raw_events and not result.grouped_segments):
        return

    rows = _result_export_rows(result)
    grouped_preview_rows, raw_preview_rows = _build_preview_rows([result])
    track_gallery_rows = _build_track_gallery_rows(grouped_preview_rows)
    raw_preview_export_rows = raw_preview_rows[:200]
    live_suffix = "_live" if live else ""
    title_suffix = " (Live)" if live else ""

    video_output_dir.mkdir(parents=True, exist_ok=True)
    (video_output_dir / f"tracking_results{live_suffix}.csv").write_text(
        _rows_to_csv(rows),
        encoding="utf-8",
    )
    _write_preview_export_directory(
        f"{result.video_name} · Grouped Preview{title_suffix}",
        grouped_preview_rows,
        video_output_dir / f"grouped_preview{live_suffix}",
    )
    _write_preview_export_directory(
        f"{result.video_name} · Raw Preview (Top 200 Rows){title_suffix}",
        raw_preview_export_rows,
        video_output_dir / f"raw_preview{live_suffix}",
    )
    _write_preview_export_directory(
        f"{result.video_name} · Track Gallery{title_suffix}",
        track_gallery_rows,
        video_output_dir / f"track_gallery{live_suffix}",
    )

    if live or not write_archives:
        return

    (video_output_dir / "tracking_grouped_preview_html.zip").write_bytes(
        _build_preview_export_zip(
            f"{result.video_name} · Grouped Preview",
            grouped_preview_rows,
            archive_root_name="tracking_grouped_preview",
        )
    )
    (video_output_dir / "tracking_raw_preview_html.zip").write_bytes(
        _build_preview_export_zip(
            f"{result.video_name} · Raw Preview (Top 200 Rows)",
            raw_preview_export_rows,
            archive_root_name="tracking_raw_preview",
        )
    )
    (video_output_dir / "tracking_track_gallery_html.zip").write_bytes(
        _build_preview_export_zip(
            f"{result.video_name} · Track Gallery",
            track_gallery_rows,
            archive_root_name="tracking_track_gallery",
        )
    )


def _queue_preview_link(video_dir: Path, candidates: list[str]) -> str:
    for candidate in candidates:
        target = video_dir / candidate
        if target.exists():
            return f"{video_dir.name}/{candidate}"
    return ""


def _write_queue_index(entries: list[dict[str, object]], run_output_dir: Path) -> None:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    body_rows: list[str] = []
    for entry in entries:
        video_dir = Path(entry["video_dir"])
        grouped_link = _queue_preview_link(
            video_dir,
            [
                "grouped_preview/index.html",
                "grouped_preview_live/index.html",
                "recovered_from_csv/grouped_preview/index.html",
                "recovered_from_crops/index.html",
            ],
        )
        raw_link = _queue_preview_link(
            video_dir,
            [
                "raw_preview/index.html",
                "raw_preview_live/index.html",
                "recovered_from_csv/raw_preview/index.html",
            ],
        )
        track_gallery_link = _queue_preview_link(
            video_dir,
            [
                "track_gallery/index.html",
                "track_gallery_live/index.html",
                "recovered_from_csv/track_gallery/index.html",
                "recovered_from_crops/track_gallery/index.html",
            ],
        )
        grouped_cell = (
            f'<a href="{html.escape(grouped_link, quote=True)}">Grouped HTML</a>'
            if grouped_link
            else '<span class="empty">—</span>'
        )
        raw_cell = (
            f'<a href="{html.escape(raw_link, quote=True)}">Raw HTML</a>'
            if raw_link
            else '<span class="empty">—</span>'
        )
        track_gallery_cell = (
            f'<a href="{html.escape(track_gallery_link, quote=True)}">Track Gallery</a>'
            if track_gallery_link
            else '<span class="empty">—</span>'
        )
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry['video_name']))}</td>"
            f"<td>{html.escape(str(entry['status']))}</td>"
            f"<td>{int(entry['frames_read'])}</td>"
            f"<td>{int(entry['processed_frames'])}</td>"
            f"<td>{int(entry['unique_track_count'])}</td>"
            f"<td>{grouped_cell}</td>"
            f"<td>{raw_cell}</td>"
            f"<td>{track_gallery_cell}</td>"
            "</tr>"
        )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Queue Tracking Index</title>
  <style>
    body {{
      margin: 24px;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f6f8fb;
      color: #172033;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #d8e0ea;
      border-radius: 16px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #e7edf5;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #eef3f9;
    }}
    .empty {{
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <h1>Queue Tracking Index</h1>
  <p>Per-video artifacts are written as each video finishes. Live previews may exist while a video is still running.</p>
  <table>
    <thead>
      <tr>
        <th>Video</th>
        <th>Status</th>
        <th>Frames Read</th>
        <th>Processed Frames</th>
        <th>Unique Tracks</th>
        <th>Grouped Preview</th>
        <th>Raw Preview</th>
        <th>Track Gallery</th>
      </tr>
    </thead>
    <tbody>
      {"".join(body_rows) if body_rows else '<tr><td colspan="8" class="empty">No processed videos yet.</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""
    (run_output_dir / "index.html").write_text(document, encoding="utf-8")


def _recover_video_output_from_crops(video_output_dir: Path) -> tuple[bool, str]:
    crops_dir = video_output_dir / "crops"
    if not crops_dir.exists():
        return False, f"No `crops/` folder found in `{video_output_dir}`."

    rows: list[dict[str, object]] = []
    for crop_path in sorted(crops_dir.iterdir()):
        if not crop_path.is_file():
            continue
        match = _TRACK_CROP_FILE_RE.match(crop_path.name)
        if match is None:
            continue
        image_id = match.group("image_id")
        track_id = int(match.group("track_id"))
        segment_index = int(match.group("segment_index"))
        rows.append(
            {
                "preview": "",
                "preview_crop_path": _format_output_path(crop_path),
                "video_name": video_output_dir.name,
                "queue_index": 1,
                "track_id": track_id,
                "segment_id": _segment_id(track_id, segment_index),
                "image_id_start": image_id,
                "image_id_end": image_id,
                "representative_image_id": image_id,
                "representative_crop_path": _format_output_path(crop_path),
                "class_name": "unknown",
                "start_time": "",
                "end_time": "",
                "duration": "",
                "hits": 1,
            }
        )

    if not rows:
        return False, f"No recoverable crop filenames found in `{crops_dir}`."

    html_path = _write_preview_export_directory(
        f"{video_output_dir.name} · Recovered Grouped Preview",
        rows,
        video_output_dir / "recovered_from_crops",
    )
    _write_preview_export_directory(
        f"{video_output_dir.name} · Recovered Track Gallery",
        _build_track_gallery_rows(rows),
        video_output_dir / "recovered_from_crops" / "track_gallery",
    )
    return True, f"Recovered HTML written to `{html_path}`."


def _recover_video_output_from_csv(video_output_dir: Path) -> tuple[bool, str]:
    csv_path = next(
        (
            candidate
            for candidate in (
                video_output_dir / "tracking_results.csv",
                video_output_dir / "tracking_results_live.csv",
            )
            if candidate.exists()
        ),
        None,
    )
    if csv_path is None:
        return False, f"No tracking CSV found in `{video_output_dir}`."

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    grouped_rows = [row for row in rows if row.get("row_type") == "grouped"]
    raw_rows = [row for row in rows if row.get("row_type") == "raw"]
    if not grouped_rows and not raw_rows:
        return False, f"Tracking CSV is empty in `{csv_path}`."

    preview_lookup = {
        (row.get("track_id", ""), row.get("segment_index", "")): {
            "crop_path": row.get("representative_crop_path", "") or "",
            "preview": "",
        }
        for row in grouped_rows
    }
    grouped_preview_rows = [
        {
            "preview": "",
            "preview_crop_path": row.get("representative_crop_path", "") or "",
            "video_name": row.get("video_name", video_output_dir.name),
            "queue_index": row.get("queue_index", ""),
            "track_id": row.get("track_id", ""),
            "segment_id": row.get("segment_id", ""),
            "class_name": row.get("class_name", ""),
            "start_time": row.get("start_time_text", ""),
            "end_time": row.get("end_time_text", ""),
            "duration": row.get("duration_text", ""),
            "start_frame": row.get("start_frame", ""),
            "end_frame": row.get("end_frame", ""),
            "image_id_start": row.get("image_id_start", ""),
            "image_id_end": row.get("image_id_end", ""),
            "representative_image_id": row.get("representative_image_id", ""),
            "representative_crop_path": row.get("representative_crop_path", ""),
            "hits": row.get("hits", ""),
        }
        for row in grouped_rows
    ]
    raw_preview_rows = [
        {
            "preview": "",
            "preview_crop_path": preview_lookup.get(
                (row.get("track_id", ""), row.get("segment_index", "")),
                {},
            ).get("crop_path", ""),
            "video_name": row.get("video_name", video_output_dir.name),
            "queue_index": row.get("queue_index", ""),
            "track_id": row.get("track_id", ""),
            "segment_id": row.get("segment_id", ""),
            "class_name": row.get("class_name", ""),
            "confidence": row.get("confidence", ""),
            "frame_num": row.get("frame_num", ""),
            "image_id": row.get("image_id", ""),
            "timestamp": row.get("timestamp_text", ""),
        }
        for row in raw_rows[:200]
    ]

    _write_preview_export_directory(
        f"{video_output_dir.name} · Recovered Grouped Preview",
        grouped_preview_rows,
        video_output_dir / "recovered_from_csv" / "grouped_preview",
    )
    _write_preview_export_directory(
        f"{video_output_dir.name} · Recovered Raw Preview (Top 200 Rows)",
        raw_preview_rows,
        video_output_dir / "recovered_from_csv" / "raw_preview",
    )
    _write_preview_export_directory(
        f"{video_output_dir.name} · Recovered Track Gallery",
        _build_track_gallery_rows(grouped_preview_rows),
        video_output_dir / "recovered_from_csv" / "track_gallery",
    )
    return True, f"Recovered HTML from `{csv_path}`."


def _recover_tracking_outputs(target_path: Path) -> tuple[bool, str]:
    if not target_path.exists():
        return False, f"Path does not exist: `{target_path}`"

    if (target_path / "crops").exists():
        ok, message = _recover_video_output_from_csv(target_path)
        if ok:
            return ok, message
        return _recover_video_output_from_crops(target_path)

    video_dirs = [path for path in target_path.iterdir() if path.is_dir() and (path / "crops").exists()]
    if not video_dirs:
        return False, f"No video output folders with `crops/` found in `{target_path}`."

    recovered = 0
    queue_index_rows: list[dict[str, object]] = []
    for video_dir in video_dirs:
        ok, _ = _recover_video_output_from_csv(video_dir)
        if not ok:
            ok, _ = _recover_video_output_from_crops(video_dir)
        if ok:
            recovered += 1
            queue_index_rows.append(
                {
                    "video_name": video_dir.name,
                    "status": "recovered",
                    "frames_read": 0,
                    "processed_frames": 0,
                    "unique_track_count": 0,
                    "video_dir": video_dir,
                }
            )
    if recovered == 0:
        return False, f"Could not rebuild any HTML files from `{target_path}`."
    if queue_index_rows:
        _write_queue_index(queue_index_rows, target_path)
    return True, f"Recovered HTML for {recovered} video folder(s) in `{target_path}`."


def _render_queue_output_summary(
    results: list[_VideoRunResult],
    run_output_dir: Path | None,
) -> None:
    if not results:
        return

    rows = [
        {
            "video_name": result.video_name,
            "status": "ok" if result.error is None else f"error: {result.error}",
            "frames_read": result.frames_read,
            "processed_frames": result.processed_frames,
            "skipped_frames": result.skipped_frames,
            "unique_tracks": result.unique_track_count,
            "output_dir": result.output_dir or "",
        }
        for result in results
    ]
    st.markdown("**Queue Output Summary**")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if run_output_dir is not None:
        st.info(
            "Per-video HTML and crop artifacts are written while the queue runs. "
            f"Root index: `{_format_output_path(run_output_dir / 'index.html')}`"
        )


def _selected_dataframe_rows(selection_event) -> list[int]:
    selection = getattr(selection_event, "selection", None)
    if selection is None and isinstance(selection_event, dict):
        selection = selection_event.get("selection")

    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows")
    if not rows:
        return []
    return [int(row_index) for row_index in rows]


def _render_preview_table(
    title: str,
    rows: list[dict[str, object]],
    *,
    key: str,
    preview_caption_columns: list[str],
) -> None:
    st.markdown(f"**{title}**")
    if not rows:
        st.info("No rows available.")
        return

    preview_column = "preview"
    df = pd.DataFrame(rows)
    if preview_column in df.columns:
        df[preview_column] = [
            str(row.get(preview_column) or "").strip()
            or _crop_path_to_data_url(_preview_crop_path_from_row(row))
            for row in rows
        ]
    display_columns = [column for column in df.columns if column != "preview_crop_path"]
    preview_available = preview_column in df.columns and df[preview_column].fillna("").astype(str).ne("").any()
    dataframe_kwargs: dict[str, object] = {
        "data": df[display_columns],
        "width": "stretch",
        "hide_index": True,
        "column_order": [preview_column] + [column for column in display_columns if column != preview_column]
        if preview_column in display_columns
        else display_columns,
        "key": key,
    }

    if preview_available:
        st.caption("Click a row to enlarge its preview below.")
        dataframe_kwargs.update(
            {
                "row_height": 92,
                "on_select": "rerun",
                "selection_mode": "single-row",
                "column_config": {
                    preview_column: st.column_config.ImageColumn("Preview", width="small")
                },
            }
        )

    selection_event = st.dataframe(**dataframe_kwargs)
    if not preview_available:
        return

    selected_rows = _selected_dataframe_rows(selection_event)
    if not selected_rows:
        return

    selected_row = df.iloc[selected_rows[0]]
    preview_data_url = str(selected_row.get(preview_column) or "")
    if not preview_data_url:
        st.info("No preview image available for the selected row.")
        return

    preview_bytes = _data_url_to_bytes(preview_data_url)
    caption_parts = [
        str(selected_row[column])
        for column in preview_caption_columns
        if column in selected_row and str(selected_row[column]).strip()
    ]
    st.image(
        preview_bytes if preview_bytes is not None else preview_data_url,
        caption=" · ".join(caption_parts) if caption_parts else None,
        width="stretch",
    )


def _render_tracking_exports(
    results: list[_VideoRunResult],
    enable_tracking: bool,
    output_dir: Path | None = None,
) -> None:
    """Show CSV downloads and preview tables for stored-video tracking runs."""
    if not enable_tracking:
        st.info("Tracking export requires `Enable Object Tracking`.")
        return

    successful = [result for result in results if result.error is None]
    total_raw = sum(len(result.raw_events) for result in successful)
    total_grouped = sum(len(result.grouped_segments) for result in successful)
    total_crops = sum(result.saved_crop_count for result in successful)

    if not successful or total_raw == 0:
        st.warning("No tracking events were recorded to export.")
        return

    all_rows = _results_export_rows(successful)
    all_csv = _rows_to_csv(all_rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tracking_results_all_videos.csv").write_text(
            all_csv,
            encoding="utf-8",
        )

    grouped_preview_rows, raw_preview_rows = _build_preview_rows(successful)
    track_gallery_rows = _build_track_gallery_rows(grouped_preview_rows)
    raw_preview_export_rows = raw_preview_rows[:200]
    grouped_preview_zip = _build_preview_export_zip(
        "Grouped Preview",
        grouped_preview_rows,
        archive_root_name="tracking_grouped_preview",
    )
    raw_preview_zip = _build_preview_export_zip(
        "Raw Preview (Top 200 Rows)",
        raw_preview_export_rows,
        archive_root_name="tracking_raw_preview",
    )
    track_gallery_zip = _build_preview_export_zip(
        "Track Gallery",
        track_gallery_rows,
        archive_root_name="tracking_track_gallery",
    )

    with st.expander("📦 Tracking Export", expanded=True):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Videos Exported", len(successful))
        metric_cols[1].metric("Raw Rows", total_raw)
        metric_cols[2].metric("Grouped Segments", total_grouped)
        metric_cols[3].metric("Saved Crops", total_crops)

        if output_dir is not None:
            st.caption(f"Local output folder: `{_format_output_path(output_dir)}`")
            (output_dir / "tracking_grouped_preview_html.zip").write_bytes(grouped_preview_zip)
            (output_dir / "tracking_raw_preview_html.zip").write_bytes(raw_preview_zip)
            (output_dir / "tracking_track_gallery_html.zip").write_bytes(track_gallery_zip)

        st.download_button(
            "⬇️ Download All Tracking CSV",
            data=all_csv,
            file_name="tracking_results_all_videos.csv",
            mime="text/csv",
            key="download_tracking_all_csv",
            on_click="ignore",
        )

        for result in successful:
            if not result.raw_events and not result.grouped_segments:
                continue
            per_video_csv = _rows_to_csv(_result_export_rows(result))
            if output_dir is not None:
                file_name = f"tracking_{_safe_path_component(result.video_name)}.csv"
                (output_dir / file_name).write_text(per_video_csv, encoding="utf-8")
            st.download_button(
                f"⬇️ Download {result.video_name} CSV",
                data=per_video_csv,
                file_name=f"tracking_{result.video_name}.csv",
                mime="text/csv",
                key=f"download_tracking_{result.queue_index}_{result.video_name}",
                on_click="ignore",
            )

        st.download_button(
            "⬇️ Download Grouped Preview HTML (.zip)",
            data=grouped_preview_zip,
            file_name="tracking_grouped_preview_html.zip",
            mime="application/zip",
            key="download_tracking_grouped_preview_zip",
            on_click="ignore",
        )

        st.download_button(
            "⬇️ Download Raw Preview HTML (.zip, top 200 rows)",
            data=raw_preview_zip,
            file_name="tracking_raw_preview_html.zip",
            mime="application/zip",
            key="download_tracking_raw_preview_zip",
            on_click="ignore",
        )

        st.download_button(
            "⬇️ Download Track Gallery HTML (.zip)",
            data=track_gallery_zip,
            file_name="tracking_track_gallery_html.zip",
            mime="application/zip",
            key="download_tracking_track_gallery_zip",
            on_click="ignore",
        )

        _render_preview_table(
            "Track Gallery",
            track_gallery_rows,
            key="tracking_track_gallery_table",
            preview_caption_columns=[
                "video_name",
                "track_id",
                "class_name",
                "first_seen",
                "last_seen",
            ],
        )

        _render_preview_table(
            "Grouped Preview",
            grouped_preview_rows,
            key="tracking_grouped_preview_table",
            preview_caption_columns=[
                "video_name",
                "segment_id",
                "class_name",
                "start_time",
                "end_time",
            ],
        )

        _render_preview_table(
            "Raw Preview (Top 200 Rows)",
            raw_preview_export_rows,
            key="tracking_raw_preview_table",
            preview_caption_columns=[
                "video_name",
                "segment_id",
                "class_name",
                "timestamp",
            ],
        )


# ── Public API ────────────────────────────────────────────────────────────────


def render(
    task: str,
    confidence: float,
    selected_model: str | None = None,
    device: str | None = None,
) -> None:
    """Render the full video-inference page for the chosen *task*."""
    st.header(f"🎬 Video · {task}")
    st.caption("Sidebar được chia theo nguồn video, tracking và hiệu năng để giảm nhầm lẫn khi cấu hình.")
    resolved_device = resolve_device(device)

    # YOLO World / YOLOE text prompt
    world_classes: list[str] | None = None
    if task == config.TASK_WORLD:
        world_classes = _world_class_input()
        if not world_classes:
            return
    elif task == config.TASK_YOLOE:
        world_classes = _yoloe_class_input()
        if not world_classes:
            return

    model = get_model_for_task(
        task,
        world_classes,
        model_name=selected_model,
        device=resolved_device,
    )
    if model is None:
        return

    # Video source
    with st.sidebar.expander("3. Source & Tracking", expanded=True):
        source = st.radio("Video Source", config.VIDEO_SOURCES, key="vid_source")
        video_processing_profile = _video_processing_profile_mode_options(source)

        # Tracking options (enabled by default)
        enable_tracking, tracker = _tracker_options()
        tracking_runtime_mode = _tracking_runtime_mode_options(enable_tracking)

        # Skip frames slider for faster inference
        skip_frames = st.slider(
            "Skip Frames",
            min_value=config.MIN_SKIP_FRAMES,
            max_value=config.MAX_SKIP_FRAMES,
            value=config.DEFAULT_SKIP_FRAMES,
            help="Process every Nth frame. Higher = faster but less smooth.",
            key="skip_frames",
        )
    perf = _performance_options(resolved_device)
    if perf.half and not _supports_half_precision(task):
        perf = replace(perf, half=False)
        st.sidebar.caption(
            "FP16 is disabled for YOLOE on this backend to avoid dtype mismatches "
            "during inference."
        )
    if resolved_device.startswith("cuda") and enable_tracking:
        st.sidebar.caption(
            "Tracking, overlay drawing, JPEG encoding and Streamlit refresh still "
            "use CPU. Disable tracking for the highest GPU utilization."
        )

    # Dispatch — pass task, world_classes & selected_model for multi-video isolation
    _SOURCE_HANDLERS[source](
        model,
        confidence,
        enable_tracking,
        tracker,
        skip_frames,
        task,
        world_classes,
        selected_model,
        resolved_device,
        perf,
        tracking_runtime_mode,
        video_processing_profile,
    )


# ── YOLOE helpers ────────────────────────────────────────────────────────────


def _yoloe_class_input() -> list[str] | None:
    """Show a text-area for category-level object classes (YOLOE)."""
    st.markdown(
        "💡 **Tip**: YOLOE supports **category-level** labels like `person`, `car`, `dog`. "
        "It does **NOT** support descriptive phrases like *person in red shirt*. "
        "Results include both bounding boxes **and** segmentation masks."
    )
    text = st.text_area(
        "🔍 Enter object categories to detect & segment in video (comma-separated)",
        value=config.DEFAULT_YOLOE_CLASSES,
        help="YOLOE will search for these object categories in every frame and produce segmentation masks.",
    )
    classes = [c.strip() for c in text.split(",") if c.strip()]
    if classes:
        st.info(f"🎯 Detecting & segmenting: **{', '.join(classes)}**")
    else:
        st.warning("⚠️ Enter at least one object category.")
    return classes or None


# ── YOLO World helpers ───────────────────────────────────────────────────────


def _world_class_input() -> list[str] | None:
    st.markdown(
        "💡 **Tip**: YOLO World v2 supports natural language! "
        "Try `person in black`, `red car`, `man with backpack`."
    )
    text = st.text_area(
        "🔍 Enter object classes or descriptions to search in video (comma-separated)",
        value=config.DEFAULT_WORLD_CLASSES,
        help="YOLO World v2 will search for these objects/descriptions in every frame.",
    )
    classes = [c.strip() for c in text.split(",") if c.strip()]
    if classes:
        st.info(f"🎯 Searching: **{', '.join(classes)}**")
    else:
        st.warning("⚠️ Enter at least one class.")
    return classes or None


# ── Tracking config ──────────────────────────────────────────────────────────


def _tracker_options() -> tuple[bool, str | None]:
    """Sidebar widgets for tracker selection."""
    enable = st.checkbox("Enable Object Tracking", value=True)
    tracker = None
    if enable:
        tracker = st.radio(
            "Tracker Algorithm",
            config.TRACKERS_LIST,
            key="tracker_algo",
        )
    return enable, tracker


def _video_processing_profile_mode_options(source: str) -> str:
    if source != config.SOURCE_STORED:
        return config.DEFAULT_VIDEO_PROCESSING_PROFILE

    default_index = config.VIDEO_PROCESSING_PROFILES.index(
        config.DEFAULT_VIDEO_PROCESSING_PROFILE
    )
    selected = st.radio(
        "Video Processing Profile",
        config.VIDEO_PROCESSING_PROFILES,
        index=default_index,
        help=(
            "Interactive keeps the current preview-first workflow. "
            "Batch Fast is tuned for long stored videos, keeps grab-skip plus no "
            "live preview, and still allows tracking exports when needed."
        ),
        key="video_processing_profile",
    )
    if selected == config.VIDEO_PROFILE_BATCH_FAST:
        st.caption(
            "Batch Fast uses grab-skip, turns off live frame rendering, and keeps "
            "tracking in fast runtime mode for better long-video throughput while "
            "still allowing crop/CSV/HTML exports."
        )
    return selected


def _video_processing_profile_options(
    video_processing_profile: str,
) -> _VideoProcessingProfileOptions:
    if video_processing_profile == config.VIDEO_PROFILE_BATCH_FAST:
        return _VideoProcessingProfileOptions(
            mode=video_processing_profile,
            render_live_preview=False,
            show_source_preview=False,
            use_grab_skip=True,
            metrics_update_interval=config.BATCH_FAST_METRICS_UPDATE_INTERVAL,
            allow_tracking_exports=True,
            force_fast_tracking=True,
        )

    return _VideoProcessingProfileOptions(
        mode=config.VIDEO_PROFILE_INTERACTIVE,
        render_live_preview=True,
        show_source_preview=True,
        use_grab_skip=False,
        metrics_update_interval=1,
        allow_tracking_exports=True,
        force_fast_tracking=False,
    )


def _tracking_runtime_mode_options(enable_tracking: bool) -> str:
    if not enable_tracking:
        return config.DEFAULT_TRACK_RUNTIME_MODE

    default_index = config.TRACK_RUNTIME_MODES.index(config.DEFAULT_TRACK_RUNTIME_MODE)
    return st.radio(
        "Tracking Runtime Mode",
        config.TRACK_RUNTIME_MODES,
        index=default_index,
        help=(
            "Fast keeps long-running streams stable by pruning live tracking state and "
            "reducing live HTML snapshot writes. Full keeps the previous unbounded behavior."
        ),
        key="tracking_runtime_mode",
    )


def _tracking_runtime_options(
    tracking_runtime_mode: str,
    *,
    has_output_dir: bool,
    force_fast: bool = False,
) -> _TrackingRuntimeOptions:
    mode = (
        tracking_runtime_mode
        if tracking_runtime_mode in config.TRACK_RUNTIME_MODES
        else config.DEFAULT_TRACK_RUNTIME_MODE
    )
    fast_mode = force_fast or mode == config.TRACK_RUNTIME_FAST
    effective_mode = config.TRACK_RUNTIME_FAST if fast_mode else mode
    # Fast mode can still keep export history when a run has an output directory.
    # The "fast" part here is the reduced preview/live snapshot behavior, not
    # dropping the final CSV/HTML artifacts for stored-video runs.
    keep_full_history = has_output_dir or not fast_mode
    return _TrackingRuntimeOptions(
        mode=effective_mode,
        keep_full_history=keep_full_history,
        live_export_interval_s=(
            config.TRACK_LIVE_EXPORT_INTERVAL_FAST_S
            if fast_mode
            else config.TRACK_LIVE_EXPORT_INTERVAL_FULL_S
        ),
        eager_live_snapshot_on_new_crop=not fast_mode,
        recent_track_window_frames=(
            None if keep_full_history else config.FAST_TRACKING_RETENTION_FRAMES
        ),
    )


# ── Frame processor ──────────────────────────────────────────────────────────


def _process_frame(
    model,
    frame: np.ndarray,
    confidence: float,
    device: str,
    perf: _PerformanceOptions,
    enable_tracking: bool,
    tracker: str | None,
    tracking_state: _TrackingMetricsState | None,
    frame_num: int,
    render_output: bool = True,
) -> tuple[np.ndarray | None, int, dict[str, int], list[_TrackObservation]]:
    """Run inference on a single frame.

    Returns ``(annotated_frame, object_count, per_class_counts)``.
    """
    if enable_tracking and tracker:
        results = model.track(
            frame,
            conf=confidence,
            device=device,
            imgsz=perf.imgsz,
            half=perf.half,
            persist=True,
            tracker=tracker,
            verbose=False,
        )
    else:
        results = model.predict(
            frame,
            conf=confidence,
            device=device,
            imgsz=perf.imgsz,
            half=perf.half,
            verbose=False,
        )

    result = results[0]
    frame_obj_count = 0
    frame_class_counts: dict[str, int] = {}
    observations: list[_TrackObservation] = []

    if result.boxes is not None and len(result.boxes):
        names = result.names
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        frame_obj_count = len(classes)

        for cls_id in classes:
            name = names[int(cls_id)]
            frame_class_counts[name] = frame_class_counts.get(name, 0) + 1

        # Accumulate unique tracked IDs
        if enable_tracking and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy()
            for track_id, cls_id, conf, box_xyxy in zip(ids, classes, confs, boxes_xyxy):
                name = names[int(cls_id)]
                if tracking_state is not None:
                    tracking_state.observe(int(track_id), name, frame_num)
                observations.append(
                    _TrackObservation(
                        track_id=int(track_id),
                        class_name=name,
                        confidence=float(conf),
                        bbox_xyxy=tuple(int(v) for v in box_xyxy),
                    )
                )

    if tracking_state is not None:
        tracking_state.prune(frame_num)

    if not render_output:
        return None, frame_obj_count, frame_class_counts, observations

    # Custom annotation with track IDs on bounding boxes
    annotated = _annotate_with_ids(frame, result, enable_tracking)

    # Overlay local + global counts
    annotated = _draw_overlay(
        annotated,
        frame_obj_count,
        frame_class_counts,
        tracking_state.tracked_total if enable_tracking and tracking_state is not None else None,
        tracking_state.class_tracked if enable_tracking and tracking_state is not None else None,
    )
    annotated = _resize_for_display(annotated)
    return annotated, frame_obj_count, frame_class_counts, observations


def _draw_overlay(
    frame: np.ndarray,
    total: int,
    class_counts: dict[str, int],
    tracked_total: int | None = None,
    class_tracked: dict[str, set[int]] | None = None,
) -> np.ndarray:
    """Draw local (per-frame) + global (cumulative) tracking overlay."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.45, 1
    y_offset = 5
    line_h = 20
    pad = 10

    lines: list[str] = []

    # ── Local (this frame) ────────────────────────────────
    local_parts = [f"In Frame: {total}"]
    for name, cnt in list(class_counts.items())[:5]:
        local_parts.append(f"{name}: {cnt}")
    lines.append(" | ".join(local_parts))

    # ── Global (cumulative tracked) ───────────────────────
    if tracked_total is not None:
        global_parts = [f"Total Tracked: {tracked_total}"]
        if class_tracked:
            for name, ids in list(class_tracked.items())[:5]:
                global_parts.append(f"{name}: {len(ids)}")
        lines.append(" | ".join(global_parts))

    # Compute box size
    max_tw = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        max_tw = max(max_tw, tw)

    box_h = y_offset + line_h * len(lines) + pad
    box_w = max_tw + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (box_w + 5, box_h + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        color = (0, 255, 0) if i == 0 else (0, 200, 255)  # green local, yellow global
        cv2.putText(
            frame,
            line,
            (pad, y_offset + line_h * (i + 1)),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return frame


# ── Sidebar live metrics ─────────────────────────────────────────────────────


class _LiveMetrics:
    """Manages sidebar placeholder widgets that update each frame."""

    def __init__(self, enable_tracking: bool):
        self.container = st.sidebar.container()
        self.enable_tracking = enable_tracking
        with self.container:
            st.subheader("📈 Live Metrics")
            self._video_ph = st.empty()
            self._elapsed_ph = st.empty()
            self._video_time_ph = st.empty()
            self._progress_ph = st.empty()
            self._frame_ph = st.empty()
            self._fps_ph = st.empty()
            st.markdown("**🟢 Local (this frame)**")
            self._objects_ph = st.empty()
            self._classes_ph = st.empty()
            if enable_tracking:
                st.markdown("**🟡 Global (cumulative)**")
                self._tracked_ph = st.empty()
                self._global_classes_ph = st.empty()

    def update(
        self,
        frame_num: int,
        frame_obj_count: int,
        frame_class_counts: dict[str, int],
        tracked_total: int,
        class_tracked: dict[str, set[int]],
        fps: float,
        elapsed_detect_s: float,
        current_video_ms: int | None,
        progress: float | None,
        video_name: str,
        queue_index: int,
        queue_total: int,
    ):
        if queue_total > 1:
            self._video_ph.markdown(
                f"**Queue** {queue_index}/{queue_total} · **Video** `{video_name}`"
            )
        else:
            self._video_ph.markdown(f"**Video** `{video_name}`")
        self._elapsed_ph.metric("Elapsed Detect Time", _format_elapsed_seconds(elapsed_detect_s))
        self._video_time_ph.metric("Current Video Time", _format_timestamp_ms(current_video_ms))
        self._progress_ph.metric("Video Progress", _format_progress(progress))
        self._frame_ph.metric("Frame", frame_num)
        self._fps_ph.metric("FPS", f"{fps:.1f}")
        self._objects_ph.metric("Objects in Frame", frame_obj_count)
        local_str = " · ".join(f"**{k}**: {v}" for k, v in frame_class_counts.items())
        self._classes_ph.markdown(local_str or "—")
        if self.enable_tracking:
            self._tracked_ph.metric("Total Unique Objects", tracked_total)
            global_str = " · ".join(
                f"**{k}**: {len(ids)}" for k, ids in class_tracked.items()
            )
            self._global_classes_ph.markdown(global_str or "—")


# ── Single-video capture loop ────────────────────────────────────────────────


def _run_video_loop(
    vid_cap: cv2.VideoCapture,
    model,
    confidence: float,
    device: str,
    perf: _PerformanceOptions,
    enable_tracking: bool,
    tracker: str | None,
    video_name: str,
    queue_index: int = 1,
    queue_total: int = 1,
    run_started_at: float | None = None,
    metrics: _LiveMetrics | None = None,
    st_frame=None,
    video_output_dir: Path | None = None,
    save_track_crops: bool = False,
    skip_frames: int = 1,
    persist_live_tracking: bool = False,
    tracking_runtime_mode: str = config.DEFAULT_TRACK_RUNTIME_MODE,
    video_processing_profile: str = config.DEFAULT_VIDEO_PROCESSING_PROFILE,
) -> _VideoRunResult:
    """Common processing loop for any ``cv2.VideoCapture`` source."""
    output_run_dir = video_output_dir.parent if video_output_dir is not None else None
    if not vid_cap.isOpened():
        st.error("❌ Could not open video source.")
        return _VideoRunResult(
            video_name=video_name,
            queue_index=queue_index,
            queue_total=queue_total,
            frames_read=0,
            processed_frames=0,
            skipped_frames=0,
            raw_events=[],
            grouped_segments=[],
            unique_track_count=0,
            output_dir=_format_output_path(output_run_dir),
            saved_crop_count=0,
            error="Could not open video source.",
        )

    processing_profile = _video_processing_profile_options(video_processing_profile)
    metrics = metrics or _LiveMetrics(enable_tracking)
    st_frame = st_frame or st.empty()
    tracking_runtime = _tracking_runtime_options(
        tracking_runtime_mode,
        has_output_dir=video_output_dir is not None,
        force_fast=processing_profile.force_fast_tracking,
    )
    persist_live_tracking = persist_live_tracking and processing_profile.allow_tracking_exports
    tracking_state = (
        _TrackingMetricsState(
            retain_all_time=tracking_runtime.keep_full_history,
            recent_window_frames=tracking_runtime.recent_track_window_frames,
        )
        if enable_tracking
        else None
    )
    store_tracking_history = (
        enable_tracking
        and processing_profile.allow_tracking_exports
        and tracking_runtime.keep_full_history
    )
    raw_events: list[_TrackingEvent] = []
    segment_runtime: dict[tuple[str, int, int], _TrackSegmentRuntime] = {}
    saved_crops: dict[tuple[str, int, int, int], _SavedCropInfo] = {}
    frame_num = 0
    processed = 0
    run_started = run_started_at or time.time()
    prev_time = time.time()
    last_bytes: bytes | None = None
    timing_meta = _capture_timing_meta(vid_cap)
    crops_dir = video_output_dir / "crops" if video_output_dir is not None else None
    last_live_export_at = time.time()
    last_live_saved_crop_count = 0

    def _write_live_tracking_snapshot(force: bool = False) -> None:
        nonlocal last_live_export_at, last_live_saved_crop_count
        if not (
            persist_live_tracking
            and enable_tracking
            and video_output_dir is not None
            and raw_events
        ):
            return
        now_ts = time.time()
        if (
            not force
            and now_ts - last_live_export_at < tracking_runtime.live_export_interval_s
            and (
                not tracking_runtime.eager_live_snapshot_on_new_crop
                or len(saved_crops) == last_live_saved_crop_count
            )
        ):
            return
        snapshot = _VideoRunResult(
            video_name=video_name,
            queue_index=queue_index,
            queue_total=queue_total,
            frames_read=frame_num,
            processed_frames=processed,
            skipped_frames=max(frame_num - processed, 0),
            raw_events=raw_events,
            grouped_segments=_group_tracking_events(raw_events, saved_crops=saved_crops),
            unique_track_count=tracking_state.tracked_total if tracking_state is not None else 0,
            output_dir=_format_output_path(output_run_dir),
            saved_crop_count=sum(1 for crop_info in saved_crops.values() if crop_info.crop_path),
        )
        _persist_video_tracking_outputs(
            snapshot,
            video_output_dir,
            live=True,
            write_archives=False,
        )
        last_live_export_at = now_ts
        last_live_saved_crop_count = len(saved_crops)

    def _process_sampled_frame(
        frame: np.ndarray,
        current_frame_num: int,
        *,
        render_frame: bool,
        update_metrics: bool,
    ) -> None:
        nonlocal last_bytes, prev_time, processed

        processed += 1
        current_video_ms = _current_video_ms(vid_cap, timing_meta, current_frame_num)
        annotated, obj_count, cls_counts, observations = _process_frame(
            model,
            frame,
            confidence,
            device,
            perf,
            enable_tracking,
            tracker,
            tracking_state,
            current_frame_num,
            render_output=render_frame,
        )
        if store_tracking_history:
            raw_events.extend(
                _build_tracking_events(
                    observations,
                    video_name=video_name,
                    queue_index=queue_index,
                    frame_num=current_frame_num,
                    timestamp_ms=current_video_ms,
                    frame=frame,
                    segment_runtime=segment_runtime,
                    saved_crops=saved_crops,
                    save_track_crops=save_track_crops,
                    crops_dir=crops_dir,
                )
            )

        if render_frame and annotated is not None:
            last_bytes = _frame_to_bytes(annotated)
            if processing_profile.render_live_preview:
                st_frame.image(last_bytes, width="stretch")

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if update_metrics:
            metrics.update(
                current_frame_num,
                obj_count,
                cls_counts,
                tracking_state.tracked_total if tracking_state is not None else 0,
                tracking_state.class_tracked if tracking_state is not None else {},
                fps,
                elapsed_detect_s=now - run_started,
                current_video_ms=current_video_ms,
                progress=_video_progress(current_video_ms, timing_meta),
                video_name=video_name,
                queue_index=queue_index,
                queue_total=queue_total,
            )

        _write_live_tracking_snapshot()

    try:
        if processing_profile.use_grab_skip:
            while vid_cap.isOpened():
                ok, frame = vid_cap.read()
                if not ok:
                    break
                frame_num += 1

                next_processed = processed + 1
                should_update_metrics = (
                    next_processed == 1
                    or next_processed % processing_profile.metrics_update_interval == 0
                )
                _process_sampled_frame(
                    frame,
                    frame_num,
                    render_frame=False,
                    update_metrics=should_update_metrics,
                )

                if skip_frames > 1:
                    frame_num += _skip_frames_by_grab(vid_cap, skip_frames - 1)
        else:
            while vid_cap.isOpened():
                ok, frame = vid_cap.read()
                if not ok:
                    break
                frame_num += 1

                if frame_num % skip_frames != 0:
                    if processing_profile.render_live_preview and last_bytes is not None:
                        st_frame.image(last_bytes, width="stretch")
                    continue

                next_processed = processed + 1
                should_render = (
                    processing_profile.render_live_preview
                    and (
                        next_processed == 1
                        or next_processed % perf.ui_update_interval == 0
                    )
                )
                _process_sampled_frame(
                    frame,
                    frame_num,
                    render_frame=should_render,
                    update_metrics=should_render,
                )
    finally:
        vid_cap.release()

    # Final summary
    skipped = frame_num - processed
    summary_parts = [f"**{frame_num}** frames read", f"**{processed}** processed"]
    if skipped:
        summary_parts.append(f"**{skipped}** skipped")

    tracked_total = tracking_state.tracked_total if tracking_state is not None else 0
    tracked_classes = tracking_state.class_tracked if tracking_state is not None else {}
    if enable_tracking and tracked_total:
        st.success(
            f"✅ **{video_name}** · {' · '.join(summary_parts)} — "
            f"**{tracked_total}** unique objects tracked"
        )
        with st.expander("📊 Tracking Summary", expanded=True):
            cols = st.columns(min(len(tracked_classes), 4) or 1)
            for idx, (name, ids) in enumerate(tracked_classes.items()):
                cols[idx % len(cols)].metric(name.capitalize(), len(ids))
    else:
        st.success(f"✅ **{video_name}** · {' · '.join(summary_parts)}")

    grouped_segments = _group_tracking_events(raw_events, saved_crops=saved_crops)
    result = _VideoRunResult(
        video_name=video_name,
        queue_index=queue_index,
        queue_total=queue_total,
        frames_read=frame_num,
        processed_frames=processed,
        skipped_frames=skipped,
        raw_events=raw_events,
        grouped_segments=grouped_segments,
        unique_track_count=tracked_total,
        output_dir=_format_output_path(output_run_dir),
        saved_crop_count=sum(1 for crop_info in saved_crops.values() if crop_info.crop_path),
    )
    if persist_live_tracking and video_output_dir is not None and enable_tracking:
        _persist_video_tracking_outputs(
            result,
            video_output_dir,
            live=True,
            write_archives=False,
        )
    return result


# ── Multi-video simultaneous loop ────────────────────────────────────────────


def _run_multi_video_loop(
    vid_names: list[str],
    videos: dict[str, object],
    confidence: float,
    device: str,
    perf: _PerformanceOptions,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
    tracking_runtime_mode: str = config.DEFAULT_TRACK_RUNTIME_MODE,
    video_processing_profile: str = config.DEFAULT_VIDEO_PROCESSING_PROFILE,
) -> None:
    """Process multiple videos simultaneously in side-by-side columns.

    Each video gets a **fresh model** instance so that ByteTrack /
    BoTSORT tracking state is isolated per video.
    """
    processing_profile = _video_processing_profile_options(video_processing_profile)
    n = len(vid_names)
    _COLS_PER_ROW = 3

    # Fresh model per video — tracking state isolation
    models = [
        load_fresh_model(task, world_classes, model_name=selected_model, device=device)
        for _ in range(n)
    ]

    # Build placeholders in a 3-per-row grid
    placeholders: list[object | None] = []
    if processing_profile.render_live_preview:
        for row_start in range(0, n, _COLS_PER_ROW):
            row_names = vid_names[row_start : row_start + _COLS_PER_ROW]
            cols = st.columns(_COLS_PER_ROW)
            for j, name in enumerate(row_names):
                with cols[j]:
                    st.markdown(f"**{name}**")
                    placeholders.append(st.empty())
    else:
        st.info("Batch Fast disables live preview while multi-video processing is running.")
        placeholders = [None for _ in vid_names]

    captures = [cv2.VideoCapture(str(videos[nm])) for nm in vid_names]
    tracking_runtime = _tracking_runtime_options(
        tracking_runtime_mode,
        has_output_dir=False,
        force_fast=processing_profile.force_fast_tracking,
    )
    tracking_states = [
        _TrackingMetricsState(
            retain_all_time=tracking_runtime.keep_full_history,
            recent_window_frames=tracking_runtime.recent_track_window_frames,
        )
        for _ in range(n)
    ]
    frame_nums = [0] * n
    processed_counts = [0] * n
    last_bytes_list: list[bytes | None] = [None] * n
    active = [cap.isOpened() for cap in captures]

    # Sidebar compact metrics
    with st.sidebar:
        st.subheader("📈 Multi-Video Metrics")
        metric_phs = [st.empty() for _ in vid_names]

    prev_time = time.time()

    try:
        while any(active):
            for i in range(n):
                if not active[i]:
                    continue

                ok, frame = captures[i].read()
                if not ok:
                    active[i] = False
                    continue

                frame_nums[i] += 1

                processed_counts[i] += 1
                if processing_profile.use_grab_skip:
                    should_render = False
                    should_update_metrics = (
                        processed_counts[i] == 1
                        or processed_counts[i] % processing_profile.metrics_update_interval == 0
                    )
                else:
                    if frame_nums[i] % skip_frames != 0:
                        processed_counts[i] -= 1
                        if (
                            processing_profile.render_live_preview
                            and last_bytes_list[i] is not None
                            and placeholders[i] is not None
                        ):
                            placeholders[i].image(
                                last_bytes_list[i],
                                width="stretch",
                            )
                        continue

                    should_render = (
                        processing_profile.render_live_preview
                        and (
                            processed_counts[i] == 1
                            or processed_counts[i] % perf.ui_update_interval == 0
                        )
                    )
                    should_update_metrics = should_render

                annotated, obj_count, cls_counts, _ = _process_frame(
                    models[i],
                    frame,
                    confidence,
                    device,
                    perf,
                    enable_tracking,
                    tracker,
                    tracking_states[i],
                    frame_nums[i],
                    render_output=should_render,
                )

                if (
                    annotated is not None
                    and processing_profile.render_live_preview
                    and placeholders[i] is not None
                ):
                    last_bytes_list[i] = _frame_to_bytes(annotated)
                    placeholders[i].image(last_bytes_list[i], width="stretch")

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                if should_update_metrics:
                    metric_phs[i].markdown(
                        f"**{vid_names[i]}** — Frame {frame_nums[i]} · "
                        f"{obj_count} obj · {tracking_states[i].tracked_total} tracked · "
                        f"{fps:.1f} FPS"
                    )

                if processing_profile.use_grab_skip and skip_frames > 1:
                    frame_nums[i] += _skip_frames_by_grab(captures[i], skip_frames - 1)
    finally:
        for cap in captures:
            cap.release()

    # Per-video summary
    for i, name in enumerate(vid_names):
        tracked_total = tracking_states[i].tracked_total
        st.success(
            f"✅ **{name}**: {frame_nums[i]} frames"
            + (f" — **{tracked_total}** unique objects" if tracked_total else "")
        )


def _run_video_queue(
    vid_names: list[str],
    videos: dict[str, object],
    confidence: float,
    device: str,
    perf: _PerformanceOptions,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
    run_output_dir: Path | None = None,
    save_track_crops: bool = False,
    tracking_runtime_mode: str = config.DEFAULT_TRACK_RUNTIME_MODE,
    video_processing_profile: str = config.DEFAULT_VIDEO_PROCESSING_PROFILE,
) -> list[_VideoRunResult]:
    """Run stored videos sequentially and collect exportable tracking results."""
    processing_profile = _video_processing_profile_options(video_processing_profile)
    results: list[_VideoRunResult] = []
    queue_index_rows: list[dict[str, object]] = []
    run_started_at = time.time()
    queue_total = len(vid_names)
    shared_frame = st.empty() if processing_profile.render_live_preview else None
    shared_metrics = _LiveMetrics(enable_tracking)

    for queue_index, name in enumerate(vid_names, start=1):
        video_dir = (
            _video_output_dir(run_output_dir, name)
            if run_output_dir is not None
            else None
        )
        processing_row_index: int | None = None
        if enable_tracking and run_output_dir is not None and video_dir is not None:
            queue_index_rows.append(
                {
                    "video_name": name,
                    "status": "processing",
                    "frames_read": 0,
                    "processed_frames": 0,
                    "unique_track_count": 0,
                    "video_dir": video_dir,
                }
            )
            processing_row_index = len(queue_index_rows) - 1
            _write_queue_index(queue_index_rows, run_output_dir)
        model = load_fresh_model(
            task,
            world_classes,
            model_name=selected_model,
            device=device,
        )
        vid_cap = cv2.VideoCapture(str(videos[name]))
        result = _run_video_loop(
            vid_cap,
            model,
            confidence,
            device,
            perf,
            enable_tracking,
            tracker,
            video_name=name,
            queue_index=queue_index,
            queue_total=queue_total,
            run_started_at=run_started_at,
            metrics=shared_metrics,
            st_frame=shared_frame,
            video_output_dir=video_dir,
            save_track_crops=save_track_crops,
            skip_frames=skip_frames,
            persist_live_tracking=enable_tracking and video_dir is not None,
            tracking_runtime_mode=tracking_runtime_mode,
            video_processing_profile=video_processing_profile,
        )
        if enable_tracking and video_dir is not None and processing_profile.allow_tracking_exports:
            _persist_video_tracking_outputs(result, video_dir, live=False, write_archives=True)
            final_row = {
                "video_name": result.video_name,
                "status": "ok" if result.error is None else f"error: {result.error}",
                "frames_read": result.frames_read,
                "processed_frames": result.processed_frames,
                "unique_track_count": result.unique_track_count,
                "video_dir": video_dir,
            }
            if processing_row_index is not None:
                queue_index_rows[processing_row_index] = final_row
            else:
                queue_index_rows.append(final_row)
            _write_queue_index(queue_index_rows, run_output_dir)
        results.append(replace(result, raw_events=[], grouped_segments=[]))

    ok_count = sum(1 for result in results if result.error is None)
    st.success(f"✅ Queue complete: **{ok_count}/{queue_total}** video(s) processed")
    failed = [result.video_name for result in results if result.error]
    if failed:
        st.warning(f"Skipped or failed videos: {', '.join(failed)}")
    return results


# ── Source handlers ──────────────────────────────────────────────────────────


def _play_stored_video(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
    device: str = config.DEVICE_CPU,
    perf: _PerformanceOptions | None = None,
    tracking_runtime_mode: str = config.DEFAULT_TRACK_RUNTIME_MODE,
    video_processing_profile: str = config.DEFAULT_VIDEO_PROCESSING_PROFILE,
) -> None:
    processing_profile = _video_processing_profile_options(video_processing_profile)
    default_video_path = _format_output_path(config.VIDEOS_DIR) or str(config.VIDEOS_DIR)
    with st.sidebar.expander("5. Stored Video Setup", expanded=True):
        selected_video_path = st.text_input(
            "Video Folder or File Path",
            value=default_video_path,
            help=(
                "Nhập đường dẫn thư mục chứa video hoặc một file video cụ thể. "
                "Đường dẫn tương đối sẽ được tính từ thư mục gốc của project."
            ),
            key="stored_video_path",
        )
    resolved_video_path = config.resolve_video_path(selected_video_path)

    # Scan the chosen path on every run so new videos appear immediately
    videos = config.get_videos_dict(selected_video_path)

    if not videos:
        if resolved_video_path.exists():
            st.warning(f"No supported videos found at `{resolved_video_path}`.")
        else:
            st.warning(f"Video path does not exist: `{resolved_video_path}`")
        return

    with st.sidebar.expander("5. Stored Video Setup", expanded=True):
        st.caption(f"Scanning videos from: `{resolved_video_path}`")
        with st.expander("Recover Tracking HTML", expanded=False):
            recover_output_path = st.text_input(
                "Tracking Output Folder",
                value="",
                help=(
                    "Nhập thư mục output của 1 video hoặc cả run queue. "
                    "Nếu app crash, hệ thống sẽ ưu tiên rebuild HTML từ CSV đã ghi; "
                    "nếu không còn CSV thì fallback sang ảnh trong folder `crops/`."
                ),
                key="recover_tracking_output_path",
            )
            if st.button("Rebuild HTML From Saved Data", key="recover_tracking_html_button"):
                target_path = _resolve_project_path(recover_output_path)
                if target_path is None:
                    st.warning("Please enter a tracking output folder path first.")
                else:
                    ok, message = _recover_tracking_outputs(target_path)
                    if ok:
                        st.success(message)
                    else:
                        st.warning(message)

        vid_names = st.multiselect(
            "Choose video(s)",
            list(videos.keys()),
            default=[list(videos.keys())[0]],
            help="Select multiple videos for simultaneous detection.",
        )
        execution_mode = st.radio(
            "Stored Video Execution",
            config.STORED_VIDEO_MODES,
            index=0,
            help=(
                "Single Video = one file. Multi Simultaneous = side-by-side. "
                "Queue Sequential = process selected videos one after another."
            ),
            key="stored_video_execution_mode",
        )
        save_track_crops = False
        if enable_tracking:
            save_track_crops = st.checkbox(
                "Save Object Crops",
                value=config.DEFAULT_SAVE_TRACK_CROPS,
                disabled=(execution_mode == config.STORED_MODE_MULTI),
                help=(
                    "Save one cropped object image for the first frame of each continuous "
                    "tracking segment. Available in Single Video and Queue Sequential mode."
                ),
                key="save_track_crops",
            )
            if execution_mode == config.STORED_MODE_MULTI:
                st.caption(
                    "Crop saving is available only in Single Video and Queue Sequential mode."
                )

    if not vid_names:
        st.info("Select at least one video from the sidebar.")
        return

    if execution_mode == config.STORED_MODE_QUEUE:
        st.markdown("**Queue Order**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "order": idx,
                        "video_name": name,
                        "path": _format_output_path(videos[name]) or str(videos[name]),
                    }
                    for idx, name in enumerate(vid_names, start=1)
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Queue mode only prepares one video at a time. The browser no longer preloads every selected file."
        )
    elif not processing_profile.show_source_preview:
        st.info(
            "Batch Fast hides the pre-run video preview to avoid browser overhead on long videos."
        )
    else:
        # ── Preview selected videos in a 3-per-row grid ──────────────
        _COLS_PER_ROW = 3
        for row_start in range(0, len(vid_names), _COLS_PER_ROW):
            row_slice = vid_names[row_start : row_start + _COLS_PER_ROW]
            cols = st.columns(_COLS_PER_ROW)
            for j, name in enumerate(row_slice):
                with cols[j]:
                    st.markdown(f"**{name}**")
                    st.video(str(videos[name]))

    with st.sidebar.expander("6. Run", expanded=True):
        run_detect = st.button("🚀 Detect Video Objects", type="primary", width="stretch")

    if run_detect:
        runtime_perf = perf or _performance_options(device)

        if execution_mode == config.STORED_MODE_SINGLE:
            if len(vid_names) != 1:
                st.sidebar.error("Single Video mode requires exactly one selected video.")
                return
            run_output_dir = (
                _create_tracking_run_dir(task)
                if enable_tracking
                else None
            )
            video_dir = (
                _video_output_dir(run_output_dir, vid_names[0])
                if run_output_dir is not None
                else None
            )
            result = _run_video_loop(
                cv2.VideoCapture(str(videos[vid_names[0]])),
                model,
                confidence,
                device,
                runtime_perf,
                enable_tracking,
                tracker,
                video_name=vid_names[0],
                video_output_dir=video_dir,
                save_track_crops=save_track_crops,
                skip_frames=skip_frames,
                persist_live_tracking=enable_tracking and video_dir is not None,
                tracking_runtime_mode=tracking_runtime_mode,
                video_processing_profile=video_processing_profile,
            )
            if enable_tracking and video_dir is not None and processing_profile.allow_tracking_exports:
                _persist_video_tracking_outputs(result, video_dir, live=False, write_archives=True)
            if processing_profile.allow_tracking_exports:
                _render_tracking_exports([result], enable_tracking, output_dir=run_output_dir)
            return

        if execution_mode == config.STORED_MODE_QUEUE:
            run_output_dir = (
                _create_tracking_run_dir(task)
                if enable_tracking
                else None
            )
            results = _run_video_queue(
                vid_names,
                videos,
                confidence,
                device,
                runtime_perf,
                enable_tracking,
                tracker,
                skip_frames,
                task,
                world_classes,
                selected_model,
                run_output_dir=run_output_dir,
                save_track_crops=save_track_crops,
                tracking_runtime_mode=tracking_runtime_mode,
                video_processing_profile=video_processing_profile,
            )
            _render_queue_output_summary(results, run_output_dir)
            return

        if len(vid_names) == 1:
            _run_video_loop(
                cv2.VideoCapture(str(videos[vid_names[0]])),
                model,
                confidence,
                device,
                runtime_perf,
                enable_tracking,
                tracker,
                video_name=vid_names[0],
                save_track_crops=False,
                skip_frames=skip_frames,
                tracking_runtime_mode=tracking_runtime_mode,
                video_processing_profile=video_processing_profile,
            )
        else:
            _run_multi_video_loop(
                vid_names,
                videos,
                confidence,
                device,
                runtime_perf,
                enable_tracking,
                tracker,
                skip_frames,
                task,
                world_classes,
                selected_model,
                tracking_runtime_mode,
                video_processing_profile,
            )
            st.info(
                "Tracking export and grouped time ranges are available in Single Video "
                "or Queue Sequential mode."
            )


def _play_webcam(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
    device: str = config.DEVICE_CPU,
    perf: _PerformanceOptions | None = None,
    tracking_runtime_mode: str = config.DEFAULT_TRACK_RUNTIME_MODE,
    video_processing_profile: str = config.DEFAULT_VIDEO_PROCESSING_PROFILE,
) -> None:
    """Browser-based webcam via streamlit-webrtc (works locally + cloud)."""
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av
    except ImportError:
        st.error(
            "❌ `streamlit-webrtc` is required for webcam access. "
            "Install it with: `pip install streamlit-webrtc`"
        )
        return

    st.info(
        "📷 Click **START** below to activate your webcam. "
        "Your browser will ask for camera permission — please allow it."
    )

    tracking_runtime = _tracking_runtime_options(
        tracking_runtime_mode,
        has_output_dir=False,
    )
    tracking_state = (
        _TrackingMetricsState(
            retain_all_time=tracking_runtime.keep_full_history,
            recent_window_frames=tracking_runtime.recent_track_window_frames,
        )
        if enable_tracking
        else None
    )
    runtime_perf = perf or _performance_options(device)

    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.processed_count = 0
            self.last_annotated = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # Skip frames
            if self.frame_count % skip_frames != 0:
                if self.last_annotated is not None:
                    return av.VideoFrame.from_ndarray(
                        self.last_annotated, format="bgr24"
                    )
                return frame

            self.processed_count += 1
            should_render = (
                self.processed_count == 1
                or self.processed_count % runtime_perf.ui_update_interval == 0
            )

            if enable_tracking and tracker:
                results = model.track(
                    img,
                    conf=confidence,
                    device=device,
                    imgsz=runtime_perf.imgsz,
                    half=runtime_perf.half,
                    persist=True,
                    tracker=tracker,
                    verbose=False,
                )
            else:
                results = model.predict(
                    img,
                    conf=confidence,
                    device=device,
                    imgsz=runtime_perf.imgsz,
                    half=runtime_perf.half,
                    verbose=False,
                )

            result = results[0]
            frame_class_counts: dict[str, int] = {}

            if result.boxes is not None and len(result.boxes):
                names = result.names
                classes = result.boxes.cls.cpu().numpy()

                for cls_id in classes:
                    name = names[int(cls_id)]
                    frame_class_counts[name] = frame_class_counts.get(name, 0) + 1

                if enable_tracking and result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy()
                    for track_id, cls_id in zip(ids, classes):
                        name = names[int(cls_id)]
                        if tracking_state is not None:
                            tracking_state.observe(int(track_id), name, self.frame_count)

            if tracking_state is not None:
                tracking_state.prune(self.frame_count)

            if not should_render and self.last_annotated is not None:
                return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")

            annotated = _annotate_with_ids(img, result, enable_tracking)
            annotated = _draw_overlay(
                annotated,
                len(result.boxes) if result.boxes is not None else 0,
                frame_class_counts,
                tracking_state.tracked_total if enable_tracking and tracking_state is not None else None,
                tracking_state.class_tracked if enable_tracking and tracking_state is not None else None,
            )
            annotated = _resize_for_display(annotated)
            self.last_annotated = annotated
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def _play_rtsp(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
    device: str = config.DEVICE_CPU,
    perf: _PerformanceOptions | None = None,
    tracking_runtime_mode: str = config.DEFAULT_TRACK_RUNTIME_MODE,
    video_processing_profile: str = config.DEFAULT_VIDEO_PROCESSING_PROFILE,
) -> None:
    with st.sidebar.expander("5. RTSP Setup", expanded=True):
        url = st.text_input(
            "RTSP Stream URL",
            placeholder="rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101",
        )
    with st.sidebar.expander("6. Run", expanded=True):
        run_stream = st.button("🚀 Start RTSP Stream", type="primary", width="stretch")
    if run_stream:
        if not url:
            st.sidebar.error("Please enter an RTSP URL.")
            return
        _run_video_loop(
            cv2.VideoCapture(url),
            model,
            confidence,
            device,
            perf or _performance_options(device),
            enable_tracking,
            tracker,
            video_name="RTSP Stream",
            skip_frames=skip_frames,
            tracking_runtime_mode=tracking_runtime_mode,
            video_processing_profile=video_processing_profile,
        )


def _play_youtube(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
    device: str = config.DEVICE_CPU,
    perf: _PerformanceOptions | None = None,
    tracking_runtime_mode: str = config.DEFAULT_TRACK_RUNTIME_MODE,
    video_processing_profile: str = config.DEFAULT_VIDEO_PROCESSING_PROFILE,
) -> None:
    with st.sidebar.expander("5. YouTube Setup", expanded=True):
        url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
        )
    with st.sidebar.expander("6. Run", expanded=True):
        run_detect = st.button("🚀 Detect YouTube Video", type="primary", width="stretch")
    if run_detect:
        if not url:
            st.sidebar.error("Please enter a YouTube URL.")
            return
        try:
            with st.sidebar:
                with st.spinner("Extracting stream URL…"):
                    stream_url = _get_youtube_stream(url)
            _run_video_loop(
                cv2.VideoCapture(stream_url),
                model,
                confidence,
                device,
                perf or _performance_options(device),
                enable_tracking,
                tracker,
                video_name="YouTube Stream",
                skip_frames=skip_frames,
                tracking_runtime_mode=tracking_runtime_mode,
                video_processing_profile=video_processing_profile,
            )
        except Exception as exc:
            st.sidebar.error(f"YouTube error: {exc}")


def _get_youtube_stream(youtube_url: str) -> str:
    ydl_opts = {"format": "best[ext=mp4]", "no_warnings": True, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


# ── Handler dispatch table ───────────────────────────────────────────────────
_SOURCE_HANDLERS = {
    config.SOURCE_STORED: _play_stored_video,
    config.SOURCE_WEBCAM: _play_webcam,
    config.SOURCE_RTSP: _play_rtsp,
    config.SOURCE_YOUTUBE: _play_youtube,
}
