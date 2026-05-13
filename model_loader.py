"""
Centralized model loading with Streamlit caching.
Models are loaded once and reused across reruns.
"""

import torch
import streamlit as st
from ultralytics import YOLO, YOLOWorld, YOLOE

import config

_CUDA_BACKENDS_CONFIGURED = False


@st.cache_resource
def load_model(model_name: str, device: str | None = None) -> YOLO:
    """Load a YOLO / RT-DETR model (detection / segmentation / pose).

    Checks the local ``weights/`` directory first; falls back to
    ultralytics auto-download, then sweeps stray weights into ``weights/``.
    """
    path = config.resolve_model_path(model_name)
    model = YOLO(path)
    _ensure_device(model, device)
    config.sweep_stray_weights()
    return model


@st.cache_resource
def load_world_model(model_name: str, device: str | None = None) -> YOLOWorld:
    """Load a YOLO World v2 model for open-vocabulary detection.

    Uses the ``YOLOWorld`` class which supports natural language
    text prompts like "person in black", "red car", etc.
    """
    path = config.resolve_model_path(model_name)
    model = YOLOWorld(path)
    _ensure_device(model, device)
    config.sweep_stray_weights()
    return model


@st.cache_resource
def load_yoloe_model(model_name: str, device: str | None = None) -> YOLOE:
    """Load a YOLOE model for text-prompted detection + segmentation.

    Uses the ``YOLOE`` class which supports category-level text prompts
    and produces both bounding boxes and instance segmentation masks.
    """
    path = config.resolve_model_path(model_name)
    model = YOLOE(path)
    _ensure_device(model, device)
    config.sweep_stray_weights()
    return model


def resolve_device(requested_device: str | None = None) -> str:
    """Resolve ``auto`` / CUDA requests to an actual runtime device string."""
    requested = requested_device or config.DEFAULT_DEVICE
    if requested == config.DEVICE_AUTO:
        return config.DEVICE_CUDA0 if torch.cuda.is_available() else config.DEVICE_CPU
    if requested == config.DEVICE_CUDA0 and not torch.cuda.is_available():
        return config.DEVICE_CPU
    return requested


def get_device_status(requested_device: str | None = None) -> dict[str, str | bool | None]:
    """Return UI-friendly runtime device information for the current host."""
    resolved = resolve_device(requested_device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    requested = requested_device or config.DEFAULT_DEVICE
    return {
        "requested": requested,
        "resolved": resolved,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "using_gpu": resolved.startswith("cuda"),
        "fallback": requested == config.DEVICE_CUDA0 and resolved == config.DEVICE_CPU,
    }


def _configure_torch_backends(device: str) -> None:
    """Enable CUDA inference settings that improve throughput on fixed-size workloads."""
    global _CUDA_BACKENDS_CONFIGURED
    if _CUDA_BACKENDS_CONFIGURED or not device.startswith("cuda"):
        return

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    _CUDA_BACKENDS_CONFIGURED = True


def _ensure_device(model, requested_device: str | None = None) -> str:
    """Move model to the best available device.

    This fixes the "index is on cpu, different from cuda:0" error
    that occurs when ``set_classes()`` creates CPU tensors while the
    model weights live on CUDA.
    """
    device = resolve_device(requested_device)
    _configure_torch_backends(device)
    try:
        # Move the inner nn.Module (catches buffers + non-parameter tensors)
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)
        model.to(device)
    except Exception:
        pass
    return device


def _set_world_classes(
    model: YOLOWorld,
    classes: list[str],
    requested_device: str | None = None,
) -> None:
    """Safely set classes on a YOLOWorld model, avoiding CPU/CUDA mismatch.

    ``set_classes()`` internally creates text-embedding tensors on CPU.
    If the model is already on CUDA (e.g. after a prior ``predict()``),
    calling ``set_classes()`` directly causes a device mismatch crash.

    Fix: move model → CPU → set_classes → move to **best** device.
    We always move to the best available device (CUDA if present) after
    setting classes, because freshly loaded models start on CPU and
    ``predict()`` alone may not move the text-embedding tensors.
    """
    # 1. CPU so set_classes creates embeddings on the same device as weights
    model.to("cpu")
    model.set_classes(classes)

    # 2. Move everything (weights + fresh text embeddings) to best device
    best = resolve_device(requested_device)
    model.to(best)


def _set_yoloe_classes(
    model: YOLOE,
    classes: list[str],
    requested_device: str | None = None,
) -> None:
    """Safely set classes on a YOLOE model, avoiding CPU/CUDA mismatch.

    Same pattern as ``_set_world_classes`` but for YOLOE.
    YOLOE supports category-level labels (not descriptive phrases).
    """
    model.to("cpu")
    model.set_classes(classes)
    best = resolve_device(requested_device)
    model.to(best)


def get_model_for_task(
    task: str,
    world_classes: list[str] | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> YOLO | YOLOWorld | YOLOE | None:
    """Return the appropriate model for *task*.

    Parameters
    ----------
    task : str
        One of the ``config.TASKS_LIST`` values.
    world_classes : list[str] | None
        Required when *task* is ``config.TASK_WORLD``.
    model_name : str | None
        Specific model filename chosen by the user via sidebar.
        Falls back to the default for the task if ``None``.
    """
    _DEFAULTS = {
        config.TASK_DETECT: config.DETECTION_MODEL,
        config.TASK_SEGMENT: config.SEGMENTATION_MODEL,
        config.TASK_POSE: config.POSE_MODEL,
        config.TASK_WORLD: config.YOLO_WORLD_MODEL,
        config.TASK_YOLOE: config.YOLOE_MODEL,
    }

    name = model_name or _DEFAULTS.get(task, config.DETECTION_MODEL)

    try:
        if task == config.TASK_WORLD:
            model = load_world_model(name, device=device)
            if world_classes:
                _set_world_classes(model, world_classes, requested_device=device)
            return model
        if task == config.TASK_YOLOE:
            model = load_yoloe_model(name, device=device)
            if world_classes:
                _set_yoloe_classes(model, world_classes, requested_device=device)
            return model
        return load_model(name, device=device)
    except Exception as exc:
        st.error(f"❌ Failed to load model for **{task}**: {exc}")
        return None


def load_fresh_model(
    task: str,
    world_classes: list[str] | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> YOLO | YOLOWorld | YOLOE:
    """Load a **fresh** (uncached) model instance.

    Used by multi-video mode so each video gets isolated tracking
    state (ByteTrack / BoTSORT state lives on the model).
    """
    _DEFAULTS = {
        config.TASK_DETECT: config.DETECTION_MODEL,
        config.TASK_SEGMENT: config.SEGMENTATION_MODEL,
        config.TASK_POSE: config.POSE_MODEL,
        config.TASK_WORLD: config.YOLO_WORLD_MODEL,
        config.TASK_YOLOE: config.YOLOE_MODEL,
    }

    name = model_name or _DEFAULTS.get(task, config.DETECTION_MODEL)
    path = config.resolve_model_path(name)

    if task == config.TASK_WORLD:
        m = YOLOWorld(path)
        config.sweep_stray_weights()
        if world_classes:
            _set_world_classes(m, world_classes, requested_device=device)
        else:
            _ensure_device(m, requested_device=device)
        return m

    if task == config.TASK_YOLOE:
        m = YOLOE(path)
        config.sweep_stray_weights()
        if world_classes:
            _set_yoloe_classes(m, world_classes, requested_device=device)
        else:
            _ensure_device(m, requested_device=device)
        return m

    m = YOLO(path)
    config.sweep_stray_weights()
    _ensure_device(m, requested_device=device)
    return m
