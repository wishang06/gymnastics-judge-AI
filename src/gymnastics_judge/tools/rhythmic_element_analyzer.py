"""
Rhythmic element analyzer (FIG split + ring, D/E scoring).
Uses only in-project engine and MediaPipe Tasks API; same logic and results as dance_judge.
"""
import asyncio
import os
from typing import Any, Dict

from .rhythmic_mediapipe_shim.mediapipe_analyzer import MediaPipeAnalyzer
from .rhythmic_engine import RhythmicAnalyzer, StrictRulesEngine


# FIG element metadata (matches pdf_rule_extractor)
RHYTHMIC_ELEMENTS = {
    "1.2096": {
        "name_en": "Switch leg deer jump with ring",
        "name_cn": "交换腿鹿跳结环",
        "difficulty_value": 0.60,
        "pdf_reference": "FIG Table #9.31",
    },
    "1.2105": {
        "name_en": "Straddle jump with ring",
        "name_cn": "跨跳结环",
        "difficulty_value": 0.50,
        "pdf_reference": "FIG Table #9.29",
    },
}


class RhythmicElementAnalyzer:
    """Analyzer for a single FIG rhythmic element (split + ring deviation, strict rules)."""

    def __init__(self, action_id: str):
        if action_id not in RHYTHMIC_ELEMENTS:
            raise ValueError(f"Unknown rhythmic element: {action_id}. Known: {list(RHYTHMIC_ELEMENTS)}")
        self.action_id = action_id
        meta = RHYTHMIC_ELEMENTS[action_id]
        self.name = f"{meta['name_en']} ({action_id})"
        self.description = (
            f"FIG rhythmic element {action_id}: {meta['name_en']}. "
            f"Split & ring deviation, D/E scoring (FIG {meta['pdf_reference']}, D={meta['difficulty_value']})."
        )
        self.video_dir = os.path.join("videos", action_id.replace(".", "_"))
        self._engine: RhythmicAnalyzer | None = None

    def _ensure_engine(self) -> RhythmicAnalyzer:
        """Lazy-build engine: MediaPipe (Tasks API) + StrictRulesEngine + RhythmicAnalyzer. No external project."""
        if self._engine is not None:
            return self._engine
        mp_analyzer = MediaPipeAnalyzer()
        rules = StrictRulesEngine()
        self._engine = RhythmicAnalyzer(mp_analyzer, rules)
        return self._engine

    async def analyze(self, video_path: str) -> Dict[str, Any]:
        """Run full analyze_video (same logic as dance_judge) in a thread; return same result dict."""
        engine = self._ensure_engine()
        return await asyncio.to_thread(engine.analyze_video, video_path)
