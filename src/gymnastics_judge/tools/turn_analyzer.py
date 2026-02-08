"""
Back Attitude Pivot (3.1203) — YOLOv8-Pose turn analyzer.
Integrates turn package (processor + evaluate) and exposes same interface as other tools.
"""
import asyncio
import os
from typing import Any, Dict

from .turn.processor import MovementProcessor
from .turn.evaluate import evaluate_turn


# FIG element metadata (Chinese name for UI)
TURN_ELEMENT = {
    "name": "后屈腿转体",
    "name_en": "Back Attitude Pivot",
    "code": "3.1203",
    "difficulty_value": 0.30,
}


class TurnAnalyzer:
    """Analyzer for Back Attitude Pivot (3.1203) using YOLOv8-Pose."""

    def __init__(self, verbose: bool = False):
        self._processor: MovementProcessor | None = None
        self.verbose = verbose
        self.name = f"{TURN_ELEMENT['name']} ({TURN_ELEMENT['code']})"
        self.description = (
            f"FIG turn element {TURN_ELEMENT['code']}: {TURN_ELEMENT['name']}. "
            f"YOLOv8-Pose rotation + thigh angle; D/E by FIG 2025-2028."
        )
        self.video_dir = os.path.join("videos", "3_1203")

    def _ensure_processor(self) -> MovementProcessor:
        if self._processor is None:
            self._processor = MovementProcessor(verbose=self.verbose)
        return self._processor

    async def analyze(self, video_path: str) -> Dict[str, Any]:
        """Run YOLO extraction then FIG scoring; return dict for pipeline (verdict, d_score, etc.)."""
        processor = self._ensure_processor()
        extracted_data = await asyncio.to_thread(
            processor.process_video, video_path, output_video_path=None
        )
        if not extracted_data:
            return {
                "verdict": "Extraction failed (no frames or YOLO error).",
                "peak_image": None,
                "d_score": 0.0,
                "e_deduction": 0.0,
                "e_reason": "No data",
                "final_score": 0.0,
                "is_d_valid": False,
                "valid_turns": 0,
                "final_official_angle": 0.0,
                "status_msg": "ERROR",
                "raw_frames": [],
            }
        eval_result = evaluate_turn(video_path, extracted_data)
        out = {
            "verdict": eval_result["report"],
            "peak_image": None,
            "raw_frames": extracted_data,
            "d_score": eval_result["d_score"],
            "e_deduction": eval_result["e_deduction"],
            "e_reason": eval_result["e_reason"],
            "final_score": eval_result["final_score"],
            "is_d_valid": eval_result["is_d_valid"],
            "valid_turns": eval_result["valid_turns"],
            "final_official_angle": eval_result["final_official_angle"],
            "status_msg": eval_result["status_msg"],
            "d_reason": eval_result["d_reason"],
            "raw_valid_degrees": eval_result["raw_valid_degrees"],
            "final_valid_degrees": eval_result["final_valid_degrees"],
        }
        return out
