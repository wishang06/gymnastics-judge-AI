# AI Gymnastics Judge System

An AI-powered system for judging gymnastics elements from video: balance (Penche), rhythmic jumps (1.2096, 1.2105), and turns (Back Attitude Pivot 3.1203). It uses **MediaPipe** and **YOLOv8-Pose** for pose/keypoints, and **Google Gemini** for rule-based scoring and Chinese reports.

## Features

- **Multiple elements**
  - **Penche (2.1106)** — Balance: split angle, 1s hold, hand support & relevé (MediaPipe + Gemini vision).
  - **交换腿鹿跳结环 (1.2096)** — Rhythmic jump: split & ring (MediaPipe).
  - **跨跳结环 (1.2105)** — Rhythmic jump: split & ring (MediaPipe).
  - **后屈腿转体 (3.1203)** — Back Attitude Pivot: rotation & thigh angle, FIG D/E (YOLOv8-Pose).
- **Reports**
  - **简易动作报告** — Short report: D/E, scoring/deduction points, one-sentence summary.
  - **综合报告** — Longer report with strengths, improvements, and next steps (optionally with video frames).
  - **运动员整体报告** — Multi-video: per-element summary + 4-dimension radar (姿态/柔韧性, 动力性, 技术规范, 稳定性) + athlete profile and practice guide.
- **Interfaces**
  - **CLI** - Tool and video selection, FIG audit output, simple & comprehensive reports.
  - **Web app** - Chinese UI: choose tool ? video ? run ? view reports; optional overall report from 2+ videos.

## Requirements

- **Python** ? 3.12  
- **MediaPipe** Pose Landmarker `.task` file (for Penche and rhythmic tools)  
- **YOLOv8-Pose** model `yolov8x-pose.pt` (for turn tool 3.1203)  
- **Google Gemini API key** (for judging and reports)

## Installation

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd gymnastics-judge-AI
```

If the repo uses **Git LFS** for the YOLO model:

```bash
git lfs install
git lfs pull
```

Install Python dependencies (with [uv](https://github.com/astral-sh/uv)):

```bash
pip install uv
uv sync
```

### 2. MediaPipe model (Penche & rhythmic)

Download the Pose Landmarker and put it in the project root:

- **Full (recommended)**: [pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)
- **Lite**: [pose_landmarker_lite.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task)

### 3. YOLO model (turn tool 3.1203)

- If you cloned with LFS and ran `git lfs pull`, `yolov8x-pose.pt` should already be in the project root.
- Otherwise, place **`yolov8x-pose.pt`** in the project root (e.g. copy from [Gymnastics_AI_Judge](https://github.com/ultralytics/ultralytics) or download via ultralytics).

### 4. Environment variables

Create a `.env` in the project root:

```bash
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Optional: `LOG_LEVEL=INFO`. You can copy from `.env.example` and edit.

## Usage

### CLI

```bash
uv run python main.py
# or
.venv\Scripts\activate   # Windows
python main.py
```

1. Choose tool (1?4): Penche, 1.2096, 1.2105, or 3.1203.  
2. Choose a video from the list.  
3. Wait for analysis; view FIG audit and simple/comprehensive reports in the terminal.

Videos are read from `videos/penche`, `videos/1_2096`, `videos/1_2105`, and `videos/3_1203`. See [VIDEO_LAYOUT.md](VIDEO_LAYOUT.md).

### Web app

From the project root (use **PowerShell or Command Prompt** on Windows, not Git Bash):

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.gymnastics_judge.web_app
```

Or use the script:

```powershell
.\run_web.ps1
```

Then open **http://localhost:5000**.

- **单次动作分析** ? Pick one tool and one video, run, then view simple and comprehensive reports.  
- **运动员整体报告** ? Pick at least two videos (any tools), run each analysis, then get the combined report with 4-dimension radar and athlete/practice text.

## Video layout

Put videos in the right folder so the app can list them:

| Folder            | Element                          |
|-------------------|----------------------------------|
| `videos/penche/`  | Penche (2.1106)                  |
| `videos/1_2096/`  | 交换腿鹿跳结环 (1.2096)          |
| `videos/1_2105/`  | 跨跳结环 (1.2105)                |
| `videos/3_1203/`  | 后屈腿转体 (3.1203)              |

Details and examples: [VIDEO_LAYOUT.md](VIDEO_LAYOUT.md).

## Project structure

```
gymnastics-judge-AI/
├── src/gymnastics_judge/
│   ├── cli.py              # CLI: tool/video selection, run, reports
│   ├── config.py           # Env and config
│   ├── core.py              # JudgeAgent, reports, vision reviews
│   ├── pipeline.py         # Single & overall analysis pipeline
│   ├── web_app.py          # Flask app and API
│   └── tools/
│       ├── pose_analyzer.py           # Penche (MediaPipe)
│       ├── rhythmic_element_analyzer.py
│       ├── rhythmic_engine.py
│       ├── rhythmic_mediapipe_shim/
│       ├── turn_analyzer.py           # Back Attitude Pivot (YOLO)
│       └── turn/                      # YOLO processor + FIG scoring
│           ├── config.py
│           ├── geometry.py
│           ├── processor.py
│           └── evaluate.py
├── static/                  # Web UI (HTML/CSS)
├── videos/                  # Per-element video folders
├── main.py                  # CLI entry
├── pyproject.toml
├── pose_landmarker_full.task # MediaPipe model
└── yolov8x-pose.pt          # YOLO model (LFS)
```

## Dependencies

- **opencv-python** — Video I/O and processing  
- **mediapipe** — Pose (Penche, rhythmic)  
- **ultralytics** — YOLOv8-Pose (turn)  
- **numpy** — Geometry and angles  
- **google-genai** — Gemini (judging and reports)  
- **flask**, **flask-cors** — Web app  
- **rich**, **typer** — CLI  
- **python-dotenv** — `.env` loading   

See [pyproject.toml](pyproject.toml) for versions.

## How it works (brief)

- **Penche**: MediaPipe pose → split angle and hold window; Gemini vision for hand support and relevé; Gemini text for D/E verdict and reports.  
- **Rhythmic (1.2096 / 1.2105)**: MediaPipe → peak frame and angles; strict FIG rules for D/E; Gemini for simple and comprehensive reports (with optional peak image).  
- **Turn (3.1203)**: YOLOv8-Pose → rotation and thigh angle; FIG rules (valid angle ≥70°, turns ≥1, buffer −150°, angle +7.5°); Gemini for Chinese reports.

Overall report: run analysis on each selected video, collect D/E and reports, then one Gemini call to produce per-element pros/cons, 4-dimension scores (1?10), athlete profile, and practice guide.