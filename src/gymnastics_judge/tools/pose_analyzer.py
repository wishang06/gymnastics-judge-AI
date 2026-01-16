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

    def __init__(self):
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
        deviation = measurements['split']['deviation']
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
        angle = np.abs(radians * 180.0 / np.pi)
        return angle

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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                measurements = self._measure_frame(landmarks, width, height)
                
                # Draw visualization
                self._draw_visualization(frame, landmarks, width, height, measurements)
                
                # Structure exactly like video_analyzer.py for consistency
                formatted_measurements = {
                    'leg_split_180': {
                        'angle_between_legs_degrees': measurements['split']['angle'],
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
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'measurements': formatted_measurements
                })
            
            # Add frame info overlay
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
            cv2.putText(frame, info_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Gymnastics Judge AI - Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return self._aggregate_results(all_measurements, fps, width, height, total_frames, duration)

    def _measure_frame(self, landmarks, width, height) -> Dict[str, Any]:
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
        angle_between_legs = self._calculate_angle(left_ankle, mid_hip, right_ankle)
        deviation = abs(180.0 - angle_between_legs)
        
        left_leg_straight = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_straight = self._calculate_angle(right_hip, right_knee, right_ankle)

        # 2. BALANCE
        lowest_ankle_y = max(left_ankle.y, right_ankle.y)
        ground_threshold = lowest_ankle_y + 0.02
        hands_touching = (left_wrist.y > ground_threshold) or (right_wrist.y > ground_threshold)

        # 3. RELEVE
        left_releve = (left_ankle.y <= left_knee.y + 0.01)
        right_releve = (right_ankle.y <= right_knee.y + 0.01)
        both_releve = left_releve and right_releve

        return {
            'split': {
                'angle': angle_between_legs,
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
        """Aggregate results with 1-second hold window logic"""
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

        # Find Peak (Max Split Angle)
        peak_frame = None
        max_split_angle = -1
        peak_idx = -1

        for i, m in enumerate(measurements):
            angle = m['measurements']['leg_split_180']['angle_between_legs_degrees']
            if angle > max_split_angle:
                max_split_angle = angle
                peak_frame = m
                peak_idx = i

        # 1-Second Window Analysis (Centered on Peak)
        hold_window_stats = None
        if peak_frame:
            half_window = int(fps / 2)
            start_idx = max(0, peak_idx - half_window)
            end_idx = min(len(measurements), peak_idx + half_window + 1)
            window = measurements[start_idx:end_idx]

            # 1. Balance: MUST be maintained throughout
            hands_touched = any(m['measurements']['balance_no_hands_ground']['any_hand_touching'] for m in window)
            
            # 2. Split: Min angle during hold
            window_angles = [m['measurements']['leg_split_180']['angle_between_legs_degrees'] for m in window]
            min_hold_angle = min(window_angles) if window_angles else 0
            
            # 3. Releve: Maintained for majority
            releve_count = sum(1 for m in window if m['measurements']['releve_tip_toed']['both_feet_releve'])
            releve_maintained = releve_count >= (len(window) / 2)

            hold_window_stats = {
                "window_start_frame": window[0]['frame'],
                "window_end_frame": window[-1]['frame'],
                "duration_frames": len(window),
                "min_split_angle_during_hold": float(min_hold_angle),
                "balance_maintained_throughout": not hands_touched,
                "releve_maintained_majority": releve_maintained
            }
            
            # Match video_analyzer.py structure exactly
            result["peak_performance"] = peak_frame  # Raw frame dict, not wrapped
            result["hold_window_1s"] = hold_window_stats  # Note: "hold_window_1s" not "hold_analysis_1s"

        return result
