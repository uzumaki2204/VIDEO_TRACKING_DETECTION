# Graph Report - yolov8-streamlit-detection-tracking  (2026-05-13)

## Corpus Check
- 9 files · ~95,079 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 236 nodes · 312 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4a794ed5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 13|Community 13]]

## God Nodes (most connected - your core abstractions)
1. `_run_video_loop()` - 19 edges
2. `🔬 YOLO Vision Studio` - 13 edges
3. `get_model_for_task()` - 11 edges
4. `_performance_options()` - 11 edges
5. `render()` - 11 edges
6. `_process_frame()` - 11 edges
7. `📖 Usage Guide` - 11 edges
8. `_ensure_device()` - 10 edges
9. `resolve_device()` - 9 edges
10. `load_fresh_model()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `render()` --calls--> `resolve_device()`  [INFERRED]
  image_service.py → model_loader.py
- `render()` --calls--> `get_model_for_task()`  [INFERRED]
  image_service.py → model_loader.py
- `render()` --calls--> `resolve_device()`  [INFERRED]
  video_service.py → model_loader.py
- `render()` --calls--> `get_model_for_task()`  [INFERRED]
  video_service.py → model_loader.py
- `_run_multi_video_loop()` --calls--> `load_fresh_model()`  [INFERRED]
  video_service.py → model_loader.py

## Communities (15 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (50): _build_tracking_events(), _capture_timing_meta(), _CaptureTimingMeta, _current_video_ms(), _default_inference_imgsz(), _default_ui_update_interval(), _finalize_segment(), _format_elapsed_seconds() (+42 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (40): _configure_torch_backends(), _ensure_device(), get_device_status(), get_model_for_task(), load_fresh_model(), load_model(), load_world_model(), load_yoloe_model() (+32 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (23): 🙏 Acknowledgements, Application Overview, 🏗️ Architecture, code:block5 (yolov8-streamlit-detection-tracking/), code:python (# Models — change to larger variants for better accuracy), code:python (# In config.py), ⚙️ Configuration, 🤝 Contributing (+15 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (23): get_device_catalog(), get_model_catalog(), get_videos_dict(), Configuration hub for YOLO Vision Studio. All paths, model configs, UI settings, Scan ``videos/`` directory each time so newly added files appear., Scan ``videos/`` directory each time so newly added files appear., Scan ``videos/`` directory each time so newly added files appear., Return local weights path if it exists, else the bare name for auto-download. (+15 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (22): _annotate_with_ids(), _color_for_track(), _draw_overlay(), _process_frame(), Draw bounding boxes with ``class | conf% | ID:N`` labels.      For segmentatio, Resize an annotated frame for UI display without shrinking inference size., Resize an annotated frame for UI display without shrinking inference size., Run inference on a single frame.      Returns ``(annotated_frame, object_count (+14 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (14): _display_results(), Image inference service — Detection, Segmentation, YOLO World & Pose.  Provide, Show a text-area for the user to type object classes / descriptive prompts., Show a text-area for the user to type object classes / descriptive prompts., Show structured results below the annotated image., Show structured results below the annotated image., Render the full image-inference page for the chosen *task*., Render the full image-inference page for the chosen *task*. (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (15): Render the full video-inference page for the chosen *task*., Show a text-area for category-level object classes (YOLOE)., Show a text-area for category-level object classes (YOLOE)., Render the full video-inference page for the chosen *task*., Sidebar widgets for tracker selection., Sidebar widgets for tracker selection., Show a text-area for category-level object classes (YOLOE)., Sidebar widgets for tracker selection. (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (11): Adding Your Own Videos, Image Inference, Image Inference, Image Inference, Image Inference, Sidebar Controls, 📖 Usage Guide, Video Inference (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.24
Nodes (10): code:bash (# 1. Clone the repository), code:powershell (powershell -ExecutionPolicy Bypass -File scripts/setup_cuda_), code:bash (# Detection), code:bash (streamlit run app.py), Download Model Weights, Installation, Prerequisites, 🚀 Quick Start (+2 more)

### Community 9 - "Community 9"
Cohesion: 0.22
Nodes (9): _frame_to_bytes(), Process multiple videos simultaneously in side-by-side columns.      Each vide, Encode a BGR *frame* to JPEG bytes for ``st.image()``.      Sending raw bytes, Encode a BGR *frame* to JPEG bytes for ``st.image()``.      Sending raw bytes a, Encode a BGR *frame* to JPEG bytes for ``st.image()``.      Sending raw bytes a, Process multiple videos simultaneously in side-by-side columns.      Each vide, Process multiple videos simultaneously in side-by-side columns.      Each vide, Process multiple videos simultaneously in side-by-side columns.      Each vide (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (5): _LiveMetrics, Manages sidebar placeholder widgets that update each frame., Manages sidebar placeholder widgets that update each frame., Manages sidebar placeholder widgets that update each frame., Manages sidebar placeholder widgets that update each frame.

## Knowledge Gaps
- **141 isolated node(s):** `YOLO Vision Studio — Main Application ====================================== R`, `Configuration hub for YOLO Vision Studio. All paths, model configs, UI settings`, `Scan ``videos/`` directory each time so newly added files appear.`, `Move any ``.pt`` files from the project root into ``weights/``.`, `Return ``{display_label: filename}`` for the given *task*.` (+136 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `render()` connect `Community 6` to `Community 0`, `Community 1`, `Community 4`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `get_model_for_task()` connect `Community 1` to `Community 5`, `Community 6`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `resolve_device()` connect `Community 1` to `Community 5`, `Community 6`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `get_model_for_task()` (e.g. with `render()` and `render()`) actually correct?**
  _`get_model_for_task()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `render()` (e.g. with `resolve_device()` and `get_model_for_task()`) actually correct?**
  _`render()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `YOLO Vision Studio — Main Application ====================================== R`, `Configuration hub for YOLO Vision Studio. All paths, model configs, UI settings`, `Scan ``videos/`` directory each time so newly added files appear.` to the rest of the system?**
  _141 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._