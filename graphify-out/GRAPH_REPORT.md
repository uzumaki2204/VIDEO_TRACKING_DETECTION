# Graph Report - yolov8-streamlit-detection-tracking  (2026-05-13)

## Corpus Check
- 9 files · ~93,636 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 186 nodes · 234 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a6850283`
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
- [[_COMMUNITY_Community 11|Community 11]]

## God Nodes (most connected - your core abstractions)
1. `🔬 YOLO Vision Studio` - 13 edges
2. `get_model_for_task()` - 11 edges
3. `_run_video_loop()` - 11 edges
4. `_ensure_device()` - 10 edges
5. `_performance_options()` - 10 edges
6. `render()` - 10 edges
7. `resolve_device()` - 9 edges
8. `_process_frame()` - 9 edges
9. `📖 Usage Guide` - 9 edges
10. `render()` - 8 edges

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

## Communities (13 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (40): _configure_torch_backends(), _ensure_device(), get_device_status(), get_model_for_task(), load_fresh_model(), load_model(), load_world_model(), load_yoloe_model() (+32 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (29): _default_inference_imgsz(), _default_ui_update_interval(), _frame_to_bytes(), _get_youtube_stream(), _LiveMetrics, _performance_options(), _PerformanceOptions, _play_rtsp() (+21 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (23): 🙏 Acknowledgements, Application Overview, 🏗️ Architecture, code:block5 (yolov8-streamlit-detection-tracking/), code:python (# Models — change to larger variants for better accuracy), code:python (# In config.py), ⚙️ Configuration, 🤝 Contributing (+15 more)

### Community 3 - "Community 3"
Cohesion: 0.1
Nodes (19): get_device_catalog(), get_model_catalog(), get_videos_dict(), Configuration hub for YOLO Vision Studio. All paths, model configs, UI settings, Scan ``videos/`` directory each time so newly added files appear., Scan ``videos/`` directory each time so newly added files appear., Return local weights path if it exists, else the bare name for auto-download., Return local weights path if it exists, else the bare name for auto-download. (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (16): _annotate_with_ids(), _color_for_track(), _draw_overlay(), _process_frame(), Resize an annotated frame for UI display without shrinking inference size., Run inference on a single frame.      Returns ``(annotated_frame, object_count, Run inference on a single frame.      Returns ``(annotated_frame, object_count, Draw local (per-frame) + global (cumulative) tracking overlay. (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (14): _display_results(), Image inference service — Detection, Segmentation, YOLO World & Pose.  Provide, Show a text-area for the user to type object classes / descriptive prompts., Show a text-area for the user to type object classes / descriptive prompts., Show structured results below the annotated image., Show structured results below the annotated image., Render the full image-inference page for the chosen *task*., Render the full image-inference page for the chosen *task*. (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.17
Nodes (12): Render the full video-inference page for the chosen *task*., Show a text-area for category-level object classes (YOLOE)., Show a text-area for category-level object classes (YOLOE)., Render the full video-inference page for the chosen *task*., Sidebar widgets for tracker selection., Sidebar widgets for tracker selection., Show a text-area for category-level object classes (YOLOE)., Sidebar widgets for tracker selection. (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.24
Nodes (10): code:bash (# 1. Clone the repository), code:powershell (powershell -ExecutionPolicy Bypass -File scripts/setup_cuda_), code:bash (# Detection), code:bash (streamlit run app.py), Download Model Weights, Installation, Prerequisites, 🚀 Quick Start (+2 more)

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (9): Adding Your Own Videos, Image Inference, Image Inference, Image Inference, Sidebar Controls, 📖 Usage Guide, Video Inference, Video Inference (+1 more)

## Knowledge Gaps
- **112 isolated node(s):** `YOLO Vision Studio — Main Application ====================================== R`, `Configuration hub for YOLO Vision Studio. All paths, model configs, UI settings`, `Scan ``videos/`` directory each time so newly added files appear.`, `Return local weights path if it exists, else the bare name for auto-download.`, `Move any ``.pt`` files from the project root into ``weights/``.` (+107 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `render()` connect `Community 6` to `Community 0`, `Community 1`, `Community 4`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `get_model_for_task()` connect `Community 0` to `Community 5`, `Community 6`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Why does `resolve_device()` connect `Community 0` to `Community 5`, `Community 6`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `get_model_for_task()` (e.g. with `render()` and `render()`) actually correct?**
  _`get_model_for_task()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `YOLO Vision Studio — Main Application ====================================== R`, `Configuration hub for YOLO Vision Studio. All paths, model configs, UI settings`, `Scan ``videos/`` directory each time so newly added files appear.` to the rest of the system?**
  _112 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._