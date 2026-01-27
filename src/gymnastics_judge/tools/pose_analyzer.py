import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import asyncio
from typing import Dict, Any, Optional

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PencheAnalyzer:
    name = "Penche Analyzer (2.1106)"
    description = "Analyzes gymnastics penche balance using MediaPipe Pose to check split angle, balance, and releve."

    def __init__(self, *, show_video: bool = True):
        self.show_video = show_video
        # Initialize MediaPipe options
        self.model_path = self._find_model()
        if not self.model_path:
            raise RuntimeError("MediaPipe Pose Landmarker model not found. Please download pose_landmarker_full.task")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # Landmarks mapping
        self.LANDMARKS = {
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24,
            'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
        }
        
        # Connections for drawing
        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Face
            (9, 10), (11, 12), (11, 9), (12, 10), # Torso
            (11, 13), (13, 15), (12, 14), (14, 16), # Arms
            (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
        ]

    def _draw_visualization(self, frame, landmarks, width, height, measurements):
        """Draw landmarks, connections, and measurements on frame"""
        # Draw connections
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                pt1 = (int(start.x * width), int(start.y * height))
                pt2 = (int(end.x * width), int(end.y * height))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks
        for lm in landmarks:
            x, y = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw Mid-Hip and Split Lines (Cyan)
        # Re-calculate mid-hip for drawing since it's not stored in 'measurements' directly as an object
        left_hip = landmarks[self.LANDMARKS['LEFT_HIP']]
        right_hip = landmarks[self.LANDMARKS['RIGHT_HIP']]
        mid_x = int((left_hip.x + right_hip.x) / 2 * width)
        mid_y = int((left_hip.y + right_hip.y) / 2 * height)
        
        left_ankle = landmarks[self.LANDMARKS['LEFT_ANKLE']]
        right_ankle = landmarks[self.LANDMARKS['RIGHT_ANKLE']]
        la_x, la_y = int(left_ankle.x * width), int(left_ankle.y * height)
        ra_x, ra_y = int(right_ankle.x * width), int(right_ankle.y * height)

        cv2.circle(frame, (mid_x, mid_y), 6, (255, 255, 0), -1)
        cv2.line(frame, (mid_x, mid_y), (la_x, la_y), (255, 255, 0), 2)
        cv2.line(frame, (mid_x, mid_y), (ra_x, ra_y), (255, 255, 0), 2)

        # Draw Stats overlay
        y_pos = height - 100
        split_angle = measurements['split']['angle']
        split_angle_raw = measurements['split'].get('angle_raw_360')
        split_angle_ext = measurements['split'].get('angle_external')
        ankle_eff = measurements['split'].get('angle_ankle')
        ankle_raw = measurements['split'].get('angle_ankle_raw_360')
        deviation = measurements['split']['deviation']
        if split_angle_raw is not None:
            extra = ""
            if ankle_eff is not None and ankle_raw is not None:
                extra = f" | ankle: {float(ankle_eff):.1f}° (raw: {float(ankle_raw):.1f}°)"
            ext_txt = f" | ext: {float(split_angle_ext):.1f}°" if split_angle_ext is not None else ""
            cv2.putText(frame, f'Split: {split_angle:.1f}° (raw: {split_angle_raw:.1f}°){ext_txt}{extra} (dev: {deviation:.1f}°)',
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f'Split: {split_angle:.1f}° (dev: {deviation:.1f}°)', 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_pos -= 30
        balanced = not measurements['balance']['hands_touching_ground']
        color = (0, 255, 0) if balanced else (0, 0, 255)
        text = "BALANCED" if balanced else "HANDS DOWN"
        cv2.putText(frame, f'Balance: {text}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_pos -= 30
        releve_status = "RELEVE" if measurements['releve']['is_releve'] else "FLAT FOOT"
        releve_color = (0, 255, 0) if measurements['releve']['is_releve'] else (0, 0, 255)
        cv2.putText(frame, f'Feet: {releve_status}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, releve_color, 2)

    def _find_model(self) -> Optional[str]:
        # Search in current dir and up one level
        paths = [
            "pose_landmarker_full.task", 
            "pose_landmarker_lite.task",
            os.path.join("..", "pose_landmarker_full.task"),
            os.path.join("src", "gymnastics_judge", "models", "pose_landmarker_full.task")
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    def _calculate_angle(self, point_a, point_b, point_c) -> float:
        a = np.array([point_a.x, point_a.y])
        b = np.array([point_b.x, point_b.y])
        c = np.array([point_c.x, point_c.y])
        
        angle_bc = np.arctan2(c[1] - b[1], c[0] - b[0])
        angle_ba = np.arctan2(a[1] - b[1], a[0] - b[0])
        radians = angle_bc - angle_ba
        angle = float(np.abs(radians * 180.0 / np.pi))
        # Normalize to [0, 180] to avoid wrap/jitter around 180°
        angle = angle % 360.0
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    def _calculate_angle_raw_360(self, point_a, point_b, point_c) -> float:
        """Raw angle in [0, 360) (no folding to <=180). Useful for debugging/UI."""
        a = np.array([point_a.x, point_a.y])
        b = np.array([point_b.x, point_b.y])
        c = np.array([point_c.x, point_c.y])

        angle_bc = np.arctan2(c[1] - b[1], c[0] - b[0])
        angle_ba = np.arctan2(a[1] - b[1], a[0] - b[0])
        radians = angle_bc - angle_ba
        angle = float(np.abs(radians * 180.0 / np.pi)) % 360.0
        return angle

    def _angle_between_vectors_3d(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Return angle in degrees between two 3D vectors."""
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0.0 or n2 == 0.0:
            return float("nan")
        cos = float(np.dot(v1, v2) / (n1 * n2))
        cos = float(np.clip(cos, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    def _world_vec(self, lm) -> np.ndarray:
        return np.array([lm.x, lm.y, lm.z], dtype=float)

    async def analyze(self, video_path: str) -> Dict[str, Any]:
        """Async wrapper for the synchronous analysis to fit the architecture"""
        return await asyncio.to_thread(self._analyze_sync, video_path)

    def _analyze_sync(self, video_path: str) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        frame_count = 0
        all_measurements = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Use container timestamp if available (more reliable than CAP_PROP_FPS)
            timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                world_landmarks = None
                if getattr(detection_result, "pose_world_landmarks", None):
                    if detection_result.pose_world_landmarks:
                        world_landmarks = detection_result.pose_world_landmarks[0]
                measurements = self._measure_frame(landmarks, world_landmarks, width, height)
                
                # Draw visualization
                self._draw_visualization(frame, landmarks, width, height, measurements)
                
                # Structure exactly like video_analyzer.py for consistency
                formatted_measurements = {
                    'leg_split_180': {
                        'angle_between_legs_degrees': measurements['split']['angle'],
                        'angle_between_legs_degrees_raw_360': measurements['split']['angle_raw_360'],
                        'angle_between_legs_degrees_external': measurements['split']['angle_external'],
                        'angle_knee_degrees': measurements['split']['angle_knee'],
                        'angle_ankle_degrees': measurements['split']['angle_ankle'],
                        'angle_knee_degrees_raw_360': measurements['split']['angle_knee_raw_360'],
                        'angle_ankle_degrees_raw_360': measurements['split']['angle_ankle_raw_360'],
                        'angle_between_legs_degrees_world_3d': measurements['split']['angle_world_3d'],
                        'deviation_from_180_degrees': measurements['split']['deviation'],
                        'left_leg_straightness_degrees': measurements['split']['left_straightness'],
                        'right_leg_straightness_degrees': measurements['split']['right_straightness']
                    },
                    'balance_no_hands_ground': {
                        'any_hand_touching': measurements['balance']['hands_touching_ground'],
                        'balance_maintained': not measurements['balance']['hands_touching_ground']
                    },
                    'releve_tip_toed': {
                        'both_feet_releve': measurements['releve']['is_releve']
                    }
                }

                all_measurements.append({
                    'frame': frame_count,
                    'timestamp': (timestamp_ms / 1000.0),
                    'timestamp_ms': timestamp_ms,
                    'measurements': formatted_measurements
                })
            
            # Add frame info overlay
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
            cv2.putText(frame, info_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.show_video:
                cv2.imshow('Gymnastics Judge AI - Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
        return self._aggregate_results(all_measurements, fps, width, height, total_frames, duration)

    def _measure_frame(self, landmarks, world_landmarks, width, height) -> Dict[str, Any]:
        """Core measurement logic using Mid-Hip"""
        get_lm = lambda name: landmarks[self.LANDMARKS[name]]
        
        left_hip = get_lm('LEFT_HIP')
        right_hip = get_lm('RIGHT_HIP')
        left_ankle = get_lm('LEFT_ANKLE')
        right_ankle = get_lm('RIGHT_ANKLE')
        left_wrist = get_lm('LEFT_WRIST')
        right_wrist = get_lm('RIGHT_WRIST')
        left_knee = get_lm('LEFT_KNEE')
        right_knee = get_lm('RIGHT_KNEE')

        # Mid-Hip Calculation
        mid_hip = Point(
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        )

        # 1. SPLIT ANGLE (Center-based)
        # Use knees for the primary split angle (typically more stable than ankles).
        angle_knee = self._calculate_angle(left_knee, mid_hip, right_knee)  # [0..180]
        angle_ankle = self._calculate_angle(left_ankle, mid_hip, right_ankle)  # [0..180]
        angle_knee_raw_360 = self._calculate_angle_raw_360(left_knee, mid_hip, right_knee)  # [0..360)
        angle_ankle_raw_360 = self._calculate_angle_raw_360(left_ankle, mid_hip, right_ankle)  # [0..360)

        angle_between_legs = angle_knee
        angle_between_legs_raw_360 = angle_knee_raw_360
        # Reflex/external representation (useful for UI expectation of ">180°")
        angle_between_legs_external = 360.0 - float(angle_between_legs)

        # Optional: 3D world estimate (debug only; can differ from 2D due to camera / z)
        angle_between_legs_world_3d = None
        if world_landmarks is not None:
            try:
                lw_hip = self._world_vec(world_landmarks[self.LANDMARKS['LEFT_HIP']])
                rw_hip = self._world_vec(world_landmarks[self.LANDMARKS['RIGHT_HIP']])
                mid_w = (lw_hip + rw_hip) / 2.0
                lw_ankle = self._world_vec(world_landmarks[self.LANDMARKS['LEFT_ANKLE']])
                rw_ankle = self._world_vec(world_landmarks[self.LANDMARKS['RIGHT_ANKLE']])
                angle_between_legs_world_3d = self._angle_between_vectors_3d(lw_ankle - mid_w, rw_ankle - mid_w)
            except Exception:
                angle_between_legs_world_3d = None

        deviation = abs(180.0 - angle_between_legs)
        
        left_leg_straight = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_straight = self._calculate_angle(right_hip, right_knee, right_ankle)

        # 2. BALANCE
        lowest_ankle_y = max(left_ankle.y, right_ankle.y)
        # Hand support heuristic:
        # - compare wrist y to the "ground" (ankle y) with a small margin
        # - require sufficient landmark visibility to avoid false positives
        ground_threshold = lowest_ankle_y + 0.02
        left_vis = float(getattr(left_wrist, "visibility", 1.0))
        right_vis = float(getattr(right_wrist, "visibility", 1.0))
        hands_touching = ((left_vis >= 0.6 and left_wrist.y > ground_threshold) or (right_vis >= 0.6 and right_wrist.y > ground_threshold))

        # 3. RELEVE
        left_releve = (left_ankle.y <= left_knee.y + 0.01)
        right_releve = (right_ankle.y <= right_knee.y + 0.01)
        both_releve = left_releve and right_releve

        return {
            'split': {
                'angle': angle_between_legs,
                'angle_raw_360': angle_between_legs_raw_360,
                'angle_world_3d': angle_between_legs_world_3d,
                'angle_external': angle_between_legs_external,
                'angle_knee': angle_knee,
                'angle_ankle': angle_ankle,
                'angle_knee_raw_360': angle_knee_raw_360,
                'angle_ankle_raw_360': angle_ankle_raw_360,
                'deviation': deviation,
                'left_straightness': left_leg_straight,
                'right_straightness': right_leg_straight
            },
            'balance': {
                'hands_touching_ground': hands_touching,
                'ground_y': lowest_ankle_y
            },
            'releve': {
                'is_releve': both_releve
            }
        }

    def _aggregate_results(self, measurements, fps, width, height, total_frames, duration):
        """Aggregate results and compute the hold segment.

        Hold timing logic (as requested):
        - Hold starts when hands are off the floor AND the split angle is held "still".
        - "Still" is defined as staying within +/- 5 degrees of the best (max) split angle
          achieved *while hands are off the floor*.
        - The hold segment is the maximal contiguous frame range around that peak that
          satisfies both conditions (hands off + within threshold).
        """
        # Default empty result structure (match video_analyzer.py exactly)
        result = {
            "video_info": {
                "resolution": f"{width}x{height}",
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": duration
            },
            "peak_performance": None,
            "hold_window_1s": None  # Match video_analyzer.py key name
        }

        if not measurements:
            return result

        def _effective_angle(raw_angle: float) -> float:
            # Convert a raw [0, 360) angle into an equivalent [0, 180] angle.
            a = float(raw_angle) % 360.0
            if a > 180.0:
                a = 360.0 - a
            return max(0.0, min(180.0, a))

        # --- Peak performance (max split) over ALL detected frames (matches previous behavior) ---
        peak_frame = None
        peak_idx = -1
        peak_eff = -1.0
        for i, m in enumerate(measurements):
            raw = m["measurements"]["leg_split_180"]["angle_between_legs_degrees"]
            if raw is None:
                continue
            eff = _effective_angle(raw)
            if eff > peak_eff:
                peak_eff = eff
                peak_frame = m
                peak_idx = i
        result["peak_performance"] = peak_frame

        # If we can't even compute a peak, we can't compute hold.
        if peak_frame is None:
            return result

        # --- Hold segment: hands OFF + angle "still" (max-min <= 5°) ---
        # This implements the user's definition more directly than anchoring to a single peak frame.
        ANGLE_STABILITY_DEG = 5.0
        # Tolerate landmark dropouts / variable decode cadence (in frames)
        MAX_MISSING_FRAME_GAP = 6

        frames = [m["frame"] for m in measurements]
        raw_angles_all = [m["measurements"]["leg_split_180"]["angle_between_legs_degrees"] for m in measurements]
        eff_angles_all = [_effective_angle(a) if a is not None else None for a in raw_angles_all]
        hands_touching_flags = [
            bool(m["measurements"]["balance_no_hands_ground"]["any_hand_touching"]) for m in measurements
        ]

        # Debounce hand-touch detection: ignore very short "touch" blips (likely landmark noise).
        # Treat a hand-touch as real only if it persists for >= 3 consecutive measurement frames.
        MIN_TOUCH_STREAK = 3
        debounced = hands_touching_flags[:]
        i = 0
        while i < len(debounced):
            if not debounced[i]:
                i += 1
                continue
            start = i
            while i < len(debounced) and debounced[i]:
                i += 1
            streak = i - start
            if streak < MIN_TOUCH_STREAK:
                for j in range(start, i):
                    debounced[j] = False
        hands_touching_flags = debounced

        # Median smoothing to reduce jitter (window size 9)
        SMOOTH_WINDOW_RADIUS = 4  # +/- 4 frames
        smoothed_eff: list[Optional[float]] = []
        for i in range(len(eff_angles_all)):
            window_vals = [
                v
                for v in eff_angles_all[
                    max(0, i - SMOOTH_WINDOW_RADIUS) : min(len(eff_angles_all), i + SMOOTH_WINDOW_RADIUS + 1)
                ]
                if v is not None
            ]
            smoothed_eff.append(float(np.median(window_vals)) if window_vals else None)

        # Build contiguous runs of indices where hands are OFF and angle is available.
        runs: list[list[int]] = []
        current: list[int] = []
        for i in range(len(measurements)):
            if hands_touching_flags[i] or smoothed_eff[i] is None:
                if current:
                    runs.append(current)
                    current = []
                continue
            if not current:
                current = [i]
                continue
            # Continue run if gap is small enough
            if (frames[i] - frames[current[-1]]) <= (1 + MAX_MISSING_FRAME_GAP):
                current.append(i)
            else:
                runs.append(current)
                current = [i]
        if current:
            runs.append(current)

        if not runs:
            result["hold_window_1s"] = {
                "window_start_frame": None,
                "window_end_frame": None,
                "duration_frames": 0,
                "duration_seconds": 0.0,
                "min_split_angle_during_hold": None,
                "min_effective_split_angle_during_hold": None,
                "balance_maintained_throughout": False,
                "releve_maintained_majority": None,
                "angle_stability_threshold_deg": float(ANGLE_STABILITY_DEG),
                "hold_reference_effective_angle_deg": None,
                "max_effective_angle_variation_within_hold_deg": None,
                "max_missing_frame_gap": MAX_MISSING_FRAME_GAP,
            }
            return result

        # Hold should be counted at/near the maximal split, but allow brief MediaPipe jitter.
        # We anchor at the best hands-off peak (by smoothed effective angle), then expand while:
        # - hands remain off the floor
        # - angle stays within +/- 5° of the reference (ref = median of top-5 hands-off angles)
        # - allowing up to 2 consecutive "bad" frames (jitter) before terminating expansion
        # Consider frames "near peak" if they are within 20° of the peak effective split.
        # This aligns with the DB validity cutoff (>=20° deviation becomes invalid).
        NEAR_PEAK_TOLERANCE_DEG = 20.0
        # Tolerate MediaPipe jitter spikes (frames that momentarily violate the +/-5° band).
        # This does NOT mean the gymnast moved; it compensates for CV noise.
        MAX_CONSEC_BAD = 20

        hands_off_idxs = [
            i for i in range(len(measurements)) if (not hands_touching_flags[i]) and (smoothed_eff[i] is not None)
        ]
        if not hands_off_idxs:
            result["hold_window_1s"] = {
                "window_start_frame": None,
                "window_end_frame": None,
                "duration_frames": 0,
                "duration_seconds": 0.0,
                "min_split_angle_during_hold": None,
                "min_effective_split_angle_during_hold": None,
                "balance_maintained_throughout": False,
                "releve_maintained_majority": None,
                "angle_stability_threshold_deg": float(ANGLE_STABILITY_DEG),
                "hold_reference_effective_angle_deg": None,
                "max_effective_angle_variation_within_hold_deg": None,
                "max_missing_frame_gap": MAX_MISSING_FRAME_GAP,
                "near_peak_tolerance_deg": float(NEAR_PEAK_TOLERANCE_DEG),
                "peak_effective_split_angle_deg": float(peak_eff),
                "max_consecutive_bad_frames": MAX_CONSEC_BAD,
            }
            return result

        # Find the best "hold" segment anywhere in the video:
        # - hands are off the floor
        # - split stays within +/- 5° of a held reference (ref chosen per-candidate)
        # - allow filling very short outlier holes (MediaPipe jitter)
        HOLE_FILL_MAX = 2

        # Build contiguous hands-off runs (break on hands-touch, missing angle, or big frame gaps)
        runs: list[list[int]] = []
        cur: list[int] = []
        for idx in range(len(measurements)):
            if hands_touching_flags[idx] or smoothed_eff[idx] is None:
                if cur:
                    runs.append(cur)
                    cur = []
                continue
            if not cur:
                cur = [idx]
                continue
            if (frames[idx] - frames[cur[-1]]) <= (1 + MAX_MISSING_FRAME_GAP):
                cur.append(idx)
            else:
                runs.append(cur)
                cur = [idx]
        if cur:
            runs.append(cur)

        # Prefer segments near the maximal split, with >=1.0s if possible.
        PEAK_INCLUSION_TOL = 5.0  # segment must reach within 5° of global peak
        best_valid = None  # (seg_max, duration_seconds, ref_eff, start_idx, end_idx)
        best_fallback = None  # (seg_max, duration_seconds, ref_eff, start_idx, end_idx)

        for run in runs:
            vals = [float(smoothed_eff[i]) for i in run]  # no None by construction
            ts = [float(measurements[i].get("timestamp_ms") or 0.0) for i in run]

            for pos in range(len(run)):
                ref = vals[pos]

                good = [abs(v - ref) <= ANGLE_STABILITY_DEG and v >= (peak_eff - NEAR_PEAK_TOLERANCE_DEG) for v in vals]
                if not good[pos]:
                    continue

                # close short holes
                closed = good[:]
                k = 0
                while k < len(closed):
                    if closed[k]:
                        k += 1
                        continue
                    hs = k
                    while k < len(closed) and not closed[k]:
                        k += 1
                    he = k - 1
                    if (
                        (he - hs + 1) <= HOLE_FILL_MAX
                        and hs - 1 >= 0
                        and he + 1 < len(closed)
                        and closed[hs - 1]
                        and closed[he + 1]
                    ):
                        for t in range(hs, he + 1):
                            closed[t] = True

                if not closed[pos]:
                    continue

                s = pos
                while s - 1 >= 0 and closed[s - 1]:
                    s -= 1
                e = pos
                while e + 1 < len(closed) and closed[e + 1]:
                    e += 1

                start_i = run[s]
                end_i = run[e]
                start_ms_c = ts[s]
                end_ms_c = ts[e]
                dur_s = (end_ms_c - start_ms_c) / 1000.0 if end_ms_c > start_ms_c else ((frames[end_i] - frames[start_i] + 1) / fps if fps > 0 else 0.0)

                seg_max = float(max(vals[s : e + 1]))
                # Must actually be near the global peak split (prevents selecting unrelated stable poses)
                if seg_max < (peak_eff - PEAK_INCLUSION_TOL):
                    continue

                cand = (seg_max, float(dur_s), float(ref), int(start_i), int(end_i))
                if float(dur_s) >= 1.0:
                    if best_valid is None or cand[0] > best_valid[0] or (cand[0] == best_valid[0] and cand[1] > best_valid[1]):
                        best_valid = cand
                else:
                    if best_fallback is None or cand[0] > best_fallback[0] or (cand[0] == best_fallback[0] and cand[1] > best_fallback[1]):
                        best_fallback = cand

        best = best_valid if best_valid is not None else best_fallback

        if best is None:
            result["hold_window_1s"] = {
                "window_start_frame": None,
                "window_end_frame": None,
                "duration_frames": 0,
                "duration_seconds": 0.0,
                "min_split_angle_during_hold": None,
                "min_effective_split_angle_during_hold": None,
                "balance_maintained_throughout": False,
                "releve_maintained_majority": None,
                "angle_stability_threshold_deg": float(ANGLE_STABILITY_DEG),
                "hold_reference_effective_angle_deg": None,
                "max_effective_angle_variation_within_hold_deg": None,
                "max_missing_frame_gap": MAX_MISSING_FRAME_GAP,
                "near_peak_tolerance_deg": float(NEAR_PEAK_TOLERANCE_DEG),
                "peak_effective_split_angle_deg": float(peak_eff),
            }
            return result

        _, duration_best_s, ref_eff, start_idx, end_idx = best

        start_frame_num = frames[start_idx]
        end_frame_num = frames[end_idx]
        hold_segment = measurements[start_idx : end_idx + 1]

        duration_frames = (end_frame_num - start_frame_num) + 1

        if not hold_segment:
            result["hold_window_1s"] = {
                "window_start_frame": int(start_frame_num),
                "window_end_frame": int(end_frame_num),
                "duration_frames": int(duration_frames),
                "duration_seconds": 0.0,
                "duration_seconds_source": "timestamp_ms",
                "min_split_angle_during_hold": None,
                "min_effective_split_angle_during_hold": None,
                "balance_maintained_throughout": True,
                "releve_maintained_majority": None,
                "angle_stability_threshold_deg": float(ANGLE_STABILITY_DEG),
                "hold_reference_effective_angle_deg": float(ref_eff),
                "max_effective_angle_variation_within_hold_deg": None,
                "max_missing_frame_gap": MAX_MISSING_FRAME_GAP,
                "near_peak_tolerance_deg": float(NEAR_PEAK_TOLERANCE_DEG),
                "peak_effective_split_angle_deg": float(peak_eff),
                "max_consecutive_bad_frames": MAX_CONSEC_BAD,
                "good_frames_used_for_stats": 0,
            }
            return result

        # Prefer real timestamps (ms) over fps math
        start_ms = float(hold_segment[0].get("timestamp_ms") or 0.0)
        end_ms = float(hold_segment[-1].get("timestamp_ms") or 0.0)
        if end_ms > start_ms:
            duration_seconds = (end_ms - start_ms) / 1000.0
            duration_source = "timestamp_ms"
        else:
            duration_seconds = (duration_frames / fps) if fps > 0 else 0.0
            duration_source = "fps"

        # Angle stats:
        # - Use SMOOTHED effective angles for judging (reduces spurious outliers).
        # - Compute the "min split" within a 1-second window around the PEAK inside the hold.
        seg_raw_angles = [m["measurements"]["leg_split_180"]["angle_between_legs_degrees"] for m in hold_segment]
        seg_eff_angles_raw = [_effective_angle(a) for a in seg_raw_angles if a is not None]
        min_eff_observed_raw = float(min(seg_eff_angles_raw)) if seg_eff_angles_raw else None

        seg_smoothed_pairs = []
        for i in range(start_idx, end_idx + 1):
            v = smoothed_eff[i]
            if v is None:
                continue
            ts = float(measurements[i].get("timestamp_ms") or 0.0)
            seg_smoothed_pairs.append((i, ts, float(v)))

        # Peak (by smoothed effective angle) inside the hold
        if seg_smoothed_pairs:
            peak_i, peak_ts, peak_v = max(seg_smoothed_pairs, key=lambda t: t[2])
        else:
            peak_i, peak_ts, peak_v = start_idx, float(hold_segment[0].get("timestamp_ms") or 0.0), float("nan")

        # 1-second window around peak (±0.5s) using timestamps
        window = [t for t in seg_smoothed_pairs if abs(t[1] - peak_ts) <= 500.0]
        if not window:
            window = seg_smoothed_pairs

        min_eff_1s = float(min(t[2] for t in window)) if window else None
        max_eff_1s = float(max(t[2] for t in window)) if window else None
        max_variation_1s = float(max_eff_1s - min_eff_1s) if (min_eff_1s is not None and max_eff_1s is not None) else None

        # Full-segment range (debug)
        min_eff_full = float(min(t[2] for t in seg_smoothed_pairs)) if seg_smoothed_pairs else None
        max_eff_full = float(max(t[2] for t in seg_smoothed_pairs)) if seg_smoothed_pairs else None

        # This is the value the judge should use:
        min_eff = min_eff_1s
        max_eff = max_eff_1s
        max_variation = max_variation_1s

        # Keep legacy field name (raw), but it now reflects the smoothed effective min in the peak ±0.5s window.
        min_raw = min_eff

        releve_true = sum(1 for m in hold_segment if m["measurements"]["releve_tip_toed"]["both_feet_releve"])
        releve_maintained = releve_true >= (len(hold_segment) / 2)

        hold_window_stats = {
            "window_start_frame": int(start_frame_num),
            "window_end_frame": int(end_frame_num),
            "duration_frames": int(duration_frames),
            "duration_seconds": float(duration_seconds),
            "duration_seconds_source": duration_source,
            "min_split_angle_during_hold": min_raw,
            "min_effective_split_angle_during_hold": min_eff,
            "min_effective_split_angle_observed_raw": min_eff_observed_raw,
            "min_effective_split_angle_full_segment": min_eff_full,
            "max_effective_split_angle_full_segment": max_eff_full,
            "peak_effective_angle_within_hold": float(peak_v) if peak_v == peak_v else None,
            "balance_maintained_throughout": True,
            "releve_maintained_majority": bool(releve_maintained),
            "angle_stability_threshold_deg": float(ANGLE_STABILITY_DEG),
            "hold_reference_effective_angle_deg": float(ref_eff),
            "max_effective_angle_variation_within_hold_deg": max_variation,
            "max_missing_frame_gap": MAX_MISSING_FRAME_GAP,
            "near_peak_tolerance_deg": float(NEAR_PEAK_TOLERANCE_DEG),
            "peak_effective_split_angle_deg": float(peak_eff),
            "max_consecutive_bad_frames": MAX_CONSEC_BAD,
            "good_frames_used_for_stats": int(len(hold_segment)),
        }

        result["hold_window_1s"] = hold_window_stats

        return result
