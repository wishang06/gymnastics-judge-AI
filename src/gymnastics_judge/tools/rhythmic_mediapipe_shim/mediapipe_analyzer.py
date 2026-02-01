"""
Shim that implements the dance_judge MediaPipeAnalyzer interface using the Tasks API
(PoseLandmarker) so rhythmic tools work without mediapipe.solutions (legacy API).
Drop-in replacement when this dir is prepended to sys.path before importing the dance project.
"""
import os
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


def _find_model() -> Optional[str]:
    # Cwd-relative (when run as "python main.py" from project root)
    cwd_paths = [
        "pose_landmarker_full.task",
        "pose_landmarker_lite.task",
        os.path.join("src", "gymnastics_judge", "models", "pose_landmarker_full.task"),
    ]
    for path in cwd_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    # Package-relative: from this file up to project root
    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            _dir = os.path.dirname(_dir)
            for name in ("pose_landmarker_full.task", "pose_landmarker_lite.task"):
                p = os.path.join(_dir, name)
                if os.path.exists(p):
                    return p
    except Exception:
        pass
    return None


# Landmark indices (same as MediaPipe 33-point pose)
NOSE = 0
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_FOOT_INDEX = 31


class MediaPipeAnalyzer:
    """Drop-in replacement for dance_judge MediaPipeAnalyzer using PoseLandmarker (Tasks API)."""

    def __init__(self):
        print("Initializing MediaPipe pose (Tasks API) for rhythmic analysis...")
        self.model_path = _find_model()
        if not self.model_path:
            raise RuntimeError(
                "Pose Landmarker model not found. Place pose_landmarker_full.task in project root."
            )
        base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print("MediaPipe (Tasks API) ready.")

    def get_video_metrics_sequence(self, video_path: str) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        sequence_data = []
        frame_idx = 0
        step = 2

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect(mp_image)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    keypoints = {}
                    for i, lm in enumerate(landmarks):
                        if getattr(lm, "visibility", 1.0) > 0.5:
                            keypoints[i] = (lm.x * w, lm.y * h)
                    if keypoints:
                        angle = self._split_angle(keypoints)
                        dev = self._ring_deviation(keypoints, h)
                        nose_y = landmarks[NOSE].y if NOSE < len(landmarks) else 0.5
                        height_score = round(1.0 - nose_y, 3)
                        sequence_data.append({
                            "f": frame_idx,
                            "a": angle,
                            "d": dev,
                            "h": height_score,
                        })
            frame_idx += 1
        cap.release()
        return sequence_data

    def _split_angle(self, keypoints: Dict[int, tuple]) -> float:
        try:
            for idx in (LEFT_KNEE, RIGHT_KNEE, LEFT_HIP, RIGHT_HIP):
                if idx not in keypoints:
                    return 0.0
            lv = np.array(keypoints[LEFT_KNEE]) - np.array(keypoints[LEFT_HIP])
            rv = np.array(keypoints[RIGHT_KNEE]) - np.array(keypoints[RIGHT_HIP])
            n = np.linalg.norm(lv) * np.linalg.norm(rv)
            if n == 0:
                return 0.0
            cos_th = np.dot(lv, rv) / n
            return round(float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))), 1)
        except Exception:
            return 0.0

    def _ring_deviation(self, keypoints: Dict[int, tuple], h: float) -> float:
        try:
            if NOSE not in keypoints or LEFT_FOOT_INDEX not in keypoints:
                return 90.0
            dist = np.linalg.norm(
                np.array(keypoints[LEFT_FOOT_INDEX]) - np.array(keypoints[NOSE])
            )
            return round(float((dist / h) * 180), 1)
        except Exception:
            return 90.0

    def measure_action(self, video_path: str, action_id: str) -> Dict[str, Any]:
        sequence = self.get_video_metrics_sequence(video_path)
        if not sequence:
            return {"status": "error"}
        peak = max(sequence, key=lambda x: x["a"])
        return {
            "status": "success",
            "split_angle": peak["a"],
            "ring_deviation_angle": peak["d"],
            "peak_frame": peak["f"],
            "sequence_count": len(sequence),
        }

    def get_landmarks_for_frame(self, rgb_frame: "np.ndarray") -> Optional[List[Any]]:
        """Return pose landmarks for a single RGB frame (list of .x, .y, .visibility) or None. Used for peak-frame extraction."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return list(result.pose_landmarks[0])
        return None
