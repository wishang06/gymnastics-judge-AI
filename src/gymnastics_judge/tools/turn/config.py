# Back Attitude Pivot (3.1203) — YOLO + FIG rules (from Gymnastics_AI_Judge)
from pathlib import Path

# Resolve project root (gymnastics-judge-AI) for model file
_TOOL_DIR = Path(__file__).resolve().parent
# src/gymnastics_judge/tools/turn -> ../../../.. = project root
_PROJECT_ROOT = _TOOL_DIR.parent.parent.parent.parent

# ------------------------------
# 1. AI engine (YOLOv8-Pose)
# ------------------------------
AI_CONFIG = {
    "ENGINE": "YOLO",
    "MODEL_PATH": str(_PROJECT_ROOT / "yolov8x-pose.pt"),
    "CONFIDENCE_THRESHOLD": 0.3,
    "DEVICE": "cpu",  # use "0" for GPU
}

# ------------------------------
# 2. Sensor thresholds
# ------------------------------
THRESHOLDS = {
    "MOVEMENT_START_ANGLE": 45.0,  # leg > 45° starts counting rotation
}

# ------------------------------
# 3. FIG 2025-2028 scoring rules
# ------------------------------
rules_config = {
    "skill_name": "Back Attitude Pivot",  # 3.1203
    "difficulty_value": 0.30,  # base for 1 turn
    "bonus_per_rotation": 0.20,  # per extra turn
    "shape_requirements": {
        "ideal_angle": 90,
        "valid_min_angle": 70,  # >= 70° valid (with deduction)
    },
    "deductions": {
        "small_deviation": 0.10,   # 80-89°
        "medium_deviation": 0.30,  # 70-79°
        "large_deviation": 0.50,   # <70° (E only; D=0)
    },
}

# Rotation buffer (degrees subtracted from raw rotation)
ROTATION_BUFFER_DEG = 150
# Angle compensation added to average thigh angle
ANGLE_COMPENSATION_DEG = 7.5
