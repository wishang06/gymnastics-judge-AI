"""
Self-contained rhythmic analysis engine: same logic and formulas as dance_judge,
using only MediaPipe Tasks API (no mediapipe.solutions). Keeps results identical.
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# FIG element difficulty (matches pdf_rule_extractor / strict_rules_engine)
ACTION_VALUE_DB = {"1.2096": 0.6, "1.2105": 0.5}


class StrictRulesEngine:
    """FIG strict rules: split/ring tolerance 20°, penalty tiers 1–10°, 10–20°, >20°. Same logic as dance_judge."""

    def __init__(self):
        self.split_standard = 180.0
        self.max_tolerance = 20.0
        self.ACTION_VALUE_DB = dict(ACTION_VALUE_DB)

    def _get_penalty_value(self, gap: float) -> float:
        if gap > 20.0:
            return 0.50
        elif gap > 10.0:
            return 0.30
        elif gap >= 1.0:
            return 0.10
        return 0.0

    def calculate_strict_score(self, measurements: Dict[str, Any], action_id: str) -> Dict[str, Any]:
        base_db_value = float(self.ACTION_VALUE_DB.get(action_id, 0.0))
        measured_split = measurements.get("split_angle", 0.0)
        split_gap = max(0.0, self.split_standard - measured_split)
        ring_dev = measurements.get("ring_deviation_angle", 0.0)
        split_valid = split_gap <= self.max_tolerance
        ring_valid = ring_dev <= self.max_tolerance
        db_admitted = split_valid and ring_valid
        total_penalty = 0.0
        e_deductions = []
        s_penalty = self._get_penalty_value(split_gap)
        if s_penalty > 0:
            total_penalty += s_penalty
            e_deductions.append(f"劈腿位置偏差 {split_gap:.1f}° : -{s_penalty:.2f}")
        r_penalty = self._get_penalty_value(ring_dev)
        if r_penalty > 0:
            total_penalty += r_penalty
            e_deductions.append(f"结环触碰偏差 {ring_dev:.1f}° : -{r_penalty:.2f}")
        return {
            "action_id": action_id,
            "raw_metrics": {"split": measured_split, "split_gap": split_gap, "ring": ring_dev},
            "db_audit": {
                "split_status": split_valid,
                "ring_status": ring_valid,
                "is_admitted": db_admitted,
                "final_db": base_db_value if db_admitted else 0.0,
            },
            "e_audit": {"total_deduction": round(total_penalty, 2), "details": e_deductions},
        }


class RhythmicAnalyzer:
    """Full analyze_video pipeline: measure_action, peak-frame extraction, strict rules, report. Uses only Tasks API."""

    def __init__(self, mediapipe_analyzer: Any, rules_engine: StrictRulesEngine):
        self.mediapipe_analyzer = mediapipe_analyzer
        self.rules_engine = rules_engine

    def _get_anatomical_hip(self, hip: Tuple[float, float], knee: Tuple[float, float]) -> np.ndarray:
        h = np.array([hip[0], hip[1]])
        k = np.array([knee[0], knee[1]])
        return h + (k - h) * 0.10

    def _calculate_improved_split_angle(
        self,
        h_l: Tuple[float, float],
        k_l: Tuple[float, float],
        h_r: Tuple[float, float],
        k_r: Tuple[float, float],
    ) -> float:
        real_h_l = self._get_anatomical_hip(h_l, k_l)
        real_h_r = self._get_anatomical_hip(h_r, k_r)
        vec_l = np.array([k_l[0] - real_h_l[0], k_l[1] - real_h_l[1]])
        vec_r = np.array([k_r[0] - real_h_r[0], k_r[1] - real_h_r[1]])
        norm_l, norm_r = np.linalg.norm(vec_l), np.linalg.norm(vec_r)
        if norm_l == 0 or norm_r == 0:
            return 180.0
        cos_theta = np.dot(vec_l, vec_r) / (norm_l * norm_r)
        raw_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return min(180.0, float(raw_angle * 1.08))

    def _get_refined_back_head_point(self, l: List[Any]) -> np.ndarray:
        if len(l) < 9:
            return np.array([0.5, 0.5])
        nose = np.array([l[0].x, l[0].y])
        ear_mid = np.array([(l[7].x + l[8].x) / 2, (l[7].y + l[8].y) / 2])
        head_dir = ear_mid - nose
        back_head = ear_mid + head_dir * 0.3
        return back_head

    def _calculate_visual_angle(
        self,
        p1: Tuple[float, float] | np.ndarray,
        p2: Tuple[float, float] | np.ndarray,
        p3: Tuple[float, float] | np.ndarray,
    ) -> float:
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 180.0
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))

    def _calculate_e_deductions(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        split_angle = metrics.get("split_angle", 180.0)
        ring_dev = metrics.get("ring_deviation_angle", 0.0)
        s_gap = max(0, 180.0 - split_angle)
        s_ded = 0.0
        if 1.0 <= s_gap <= 10.0:
            s_ded = 0.1
        elif 10.0 < s_gap <= 20.0:
            s_ded = 0.3
        elif s_gap > 20.0:
            s_ded = 0.5
        r_ded = 0.0
        if 1.0 <= ring_dev <= 10.0:
            r_ded = 0.1
        elif 10.0 < ring_dev <= 20.0:
            r_ded = 0.3
        elif ring_dev > 20.0:
            r_ded = 0.5
        return {
            "split_gap": round(s_gap, 1),
            "ring_gap": round(ring_dev, 1),
            "split_deduction": s_ded,
            "ring_deduction": r_ded,
            "total_deduction": round(s_ded + r_ded, 2),
        }

    def _extract_peak_frame(
        self, video_path: str, frame_index: int, action_id: str
    ) -> Tuple[Optional[str], float, float]:
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            best_frame_data: Optional[Any] = None
            best_landmarks: Optional[List[Any]] = None
            max_score = -1.0
            actual_idx = frame_index

            search_range = range(max(0, frame_index - 10), min(total_frames, frame_index + 11))
            for f_idx in search_range:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                l = self.mediapipe_analyzer.get_landmarks_for_frame(rgb)
                if l is None or len(l) < 33:
                    continue
                try:
                    vis = np.mean([getattr(l[i], "visibility", 1.0) for i in [23, 24, 25, 26, 31, 32]])
                except (IndexError, TypeError):
                    continue
                try:
                    foot_dist = np.sqrt((l[31].x - l[32].x) ** 2 + (l[31].y - l[32].y) ** 2)
                except (IndexError, AttributeError, TypeError):
                    continue
                overlap_penalty = 100 if foot_dist < 0.05 else 0
                height_score = (1.0 - min(l[31].y, l[32].y)) * 30
                current_score = (vis * 100) + (foot_dist * 50) + height_score - overlap_penalty
                try:
                    if l[31].y > (l[23].y + l[24].y) / 2 and l[32].y > (l[23].y + l[24].y) / 2:
                        continue
                except (IndexError, AttributeError, TypeError):
                    pass
                    continue
                if current_score > max_score:
                    max_score = current_score
                    best_frame_data = frame.copy()
                    best_landmarks = l
                    actual_idx = f_idx
            cap.release()

            if best_frame_data is not None and best_landmarks is not None:
                try:
                    frame = best_frame_data
                    h, w = frame.shape[0], frame.shape[1]
                    l = best_landmarks
                    if len(l) < 33:
                        return None, -1.0, -1.0
                    rh_l = self._get_anatomical_hip((l[23].x, l[23].y), (l[25].x, l[25].y))
                    rh_r = self._get_anatomical_hip((l[24].x, l[24].y), (l[26].x, l[26].y))
                    bh_pt = self._get_refined_back_head_point(l)
                    hip_mid = (rh_l + rh_r) / 2

                    def to_px(pt: Any) -> Tuple[int, int]:
                        return (int(float(pt[0]) * w), int(float(pt[1]) * h))

                    p_head_px = to_px(bh_pt)
                    dist_l = np.linalg.norm(np.array(p_head_px) - np.array(to_px((l[31].x, l[31].y))))
                    dist_r = np.linalg.norm(np.array(p_head_px) - np.array(to_px((l[32].x, l[32].y))))
                    back_idx = 31 if dist_l < dist_r else 32

                    cv2.line(frame, to_px(rh_l), to_px((l[25].x, l[25].y)), (0, 255, 0), 5)
                    cv2.line(frame, to_px(rh_r), to_px((l[26].x, l[26].y)), (0, 255, 0), 5)
                    cv2.line(frame, to_px(bh_pt), to_px(hip_mid), (0, 0, 255), 5)
                    cv2.line(frame, to_px(hip_mid), to_px((l[back_idx].x, l[back_idx].y)), (0, 0, 255), 5)
                    cur_split = self._calculate_improved_split_angle(
                        (l[23].x, l[23].y), (l[25].x, l[25].y), (l[24].x, l[24].y), (l[26].x, l[26].y)
                    )
                    cur_ring = self._calculate_visual_angle(bh_pt, hip_mid, (l[back_idx].x, l[back_idx].y))
                    cv2.putText(frame, f"SPLIT: {cur_split:.1f}", (30, 60), 2, 1.2, (0, 255, 0), 2)
                    cv2.putText(frame, f"RING: {cur_ring:.1f}", (30, 110), 2, 1.2, (0, 0, 255), 2)
                    # One file per (video, element): same run overwrites previous
                    audit_dir = os.path.join(os.getcwd(), "audit_frames")
                    os.makedirs(audit_dir, exist_ok=True)
                    video_stem = os.path.splitext(os.path.basename(video_path))[0]
                    safe_stem = re.sub(r"[^\w\-.]", "_", video_stem) or "video"
                    action_safe = action_id.replace(".", "_")
                    out_name = os.path.join(audit_dir, f"{safe_stem}_{action_safe}.jpg")
                    cv2.imwrite(out_name, frame)
                    return out_name, cur_ring, cur_split
                except (IndexError, AttributeError, TypeError, OSError) as e:
                    print(f"Peak frame write/landmark error: {e}")
                    return None, -1.0, -1.0
        except Exception as e:
            print(f"Error: {e}")
        return None, -1.0, -1.0

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        filename = os.path.basename(video_path)
        id_match = re.search(r"(\d)(\d{4})", filename)
        declared_id = f"{id_match.group(1)}.{id_match.group(2)}" if id_match else "1.2096"
        raw = self.mediapipe_analyzer.measure_action(video_path, declared_id)
        img_path, p_ring, p_split = self._extract_peak_frame(
            video_path, raw.get("peak_frame", 0), declared_id
        )
        if p_split > 0:
            raw["split_angle"] = p_split
        if p_ring > 0:
            raw["ring_deviation_angle"] = p_ring
        audit_detail = self.rules_engine.calculate_strict_score(raw, declared_id)
        e_info = self._calculate_e_deductions(raw)
        return {
            "action_id": declared_id,
            "measurements": raw,
            "is_valid": audit_detail.get("db_audit", {}).get("is_admitted", False),
            "difficulty_score": audit_detail.get("db_audit", {}).get("final_db", 0.0),
            "e_deductions": e_info,
            "peak_image": img_path,
            "score_text": self._format_report(audit_detail, raw, e_info),
        }

    def _format_report(
        self, audit: Dict[str, Any], metrics: Dict[str, Any], e_info: Dict[str, Any]
    ) -> str:
        is_valid = audit.get("db_audit", {}).get("is_admitted", False)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        return f"""
====================================
      AI 几何审计报告 (含E分扣分)
====================================
结果: {status} | 难度分 (D): {audit.get('db_audit', {}).get('final_db', 0.0)}
------------------------------------
[实测数据]
劈叉开度: {metrics.get('split_angle', 0):.1f}° (偏差: {e_info['split_gap']}°)
结环偏差: {metrics.get('ring_deviation_angle', 0):.1f}°
------------------------------------
[E分扣分审计]
- 劈叉偏差扣分: -{e_info['split_deduction']:.2f}
- 结环位置扣分: -{e_info['ring_deduction']:.2f}
- 技术扣分总计: -{e_info['total_deduction']:.2f}
===================================="""
