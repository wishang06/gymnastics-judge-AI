"""
YOLOv8-Pose video processor for Back Attitude Pivot (3.1203).
Per-frame rotation + thigh angle; inertia and sample-and-hold logic preserved.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

from . import config
from .geometry import calculate_azimuth


class MovementProcessor:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        model_path = config.AI_CONFIG["MODEL_PATH"]
        if self.verbose:
            print(f"Loading YOLOv8-Pose: {model_path} ...")
        self.model = YOLO(model_path)
        self.device = config.AI_CONFIG["DEVICE"]
        self.conf_thres = config.AI_CONFIG["CONFIDENCE_THRESHOLD"]

    def process_video(self, video_path: str, output_video_path: str | None = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if self.verbose:
                print(f"Error: cannot open video {video_path}")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        extracted_data = []
        prev_azimuth = 0
        cumulative_rotation = 0
        rotation_direction = 0
        current_velocity = 0.0
        INERTIA_DECAY = 0.98
        last_valid_angle = 0.0
        is_first_frame = True

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(
                frame, device=self.device, conf=self.conf_thres, verbose=False
            )

            frame_data = {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "is_detected": False,
                "total_rotation": 0,
                "thigh_lift_angle": 0,
                "is_releve": True,
            }

            detected_this_frame = False

            if len(results) > 0 and results[0].keypoints is not None:
                kpts = results[0].keypoints.data[0].cpu().numpy()
                if np.sum(kpts[[5, 6, 11, 12], 2] > 0.3) == 4:
                    detected_this_frame = True
                    frame_data["is_detected"] = True

                    def get_pt(idx):
                        return np.array([kpts[idx][0], kpts[idx][1], 0])

                    azimuth = calculate_azimuth(get_pt(5), get_pt(6))
                    if is_first_frame:
                        prev_azimuth = azimuth
                        is_first_frame = False
                        delta = 0
                    else:
                        delta = azimuth - prev_azimuth
                        if delta < -180:
                            delta += 360
                        elif delta > 180:
                            delta -= 360
                        if rotation_direction == 0 and abs(delta) > 5:
                            rotation_direction = np.sign(delta)
                        if rotation_direction != 0:
                            if np.sign(delta) != rotation_direction:
                                fix = delta + (180 * rotation_direction)
                                if (
                                    np.sign(fix) == rotation_direction
                                    and abs(fix) < 120
                                ):
                                    delta = fix
                                else:
                                    delta = current_velocity
                            if abs(delta) > 100:
                                delta = current_velocity
                        if abs(delta) < 100:
                            current_velocity = 0.7 * delta + 0.3 * current_velocity
                        cumulative_rotation += delta
                        prev_azimuth = azimuth

                    mid_shoulder = (get_pt(5) + get_pt(6)) / 2
                    mid_hip = (get_pt(11) + get_pt(12)) / 2
                    torso_len = np.linalg.norm(mid_shoulder - mid_hip)
                    LENGTH_THRESHOLD = torso_len * 0.5
                    vertical_vec = np.array([0, 1, 0])

                    def calculate_valid_leg_angle(h_idx, k_idx):
                        if kpts[h_idx][2] < 0.5 or kpts[k_idx][2] < 0.5:
                            return None
                        vec = get_pt(k_idx) - get_pt(h_idx)
                        proj_len = np.linalg.norm(vec)
                        if proj_len < LENGTH_THRESHOLD:
                            return None
                        angle = np.degrees(
                            np.arccos(
                                np.clip(
                                    np.dot(vec, vertical_vec) / proj_len, -1, 1
                                )
                            )
                        )
                        if angle <= 60.0:
                            return None
                        return angle

                    valid_l = calculate_valid_leg_angle(11, 13)
                    valid_r = calculate_valid_leg_angle(12, 14)
                    current_best_angle = 0
                    if valid_l is not None and valid_r is not None:
                        current_best_angle = max(valid_l, valid_r)
                    elif valid_l is not None:
                        current_best_angle = valid_l
                    elif valid_r is not None:
                        current_best_angle = valid_r

                    if current_best_angle > 0:
                        last_valid_angle = current_best_angle
                    frame_data["thigh_lift_angle"] = last_valid_angle

                    if out:
                        res_plotted = results[0].plot()
                        out.write(res_plotted)

            if not detected_this_frame and rotation_direction != 0:
                cumulative_rotation += current_velocity
                current_velocity *= INERTIA_DECAY
                prev_azimuth = (prev_azimuth + current_velocity) % 360

            frame_data["total_rotation"] = cumulative_rotation
            extracted_data.append(frame_data)
            if out and not detected_this_frame:
                out.write(frame)
            frame_idx += 1

        cap.release()
        if out:
            out.release()
        return extracted_data
