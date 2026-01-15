"""
Video Analysis Tool - MediaPipe Pose Detection & Angle Measurement
Analyzes local .mp4 video files and calculates joint angles
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import sys
import argparse
import json

# Function to calculate distance between two points
def calculate_distance(point_a, point_b):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point_a: First point (MediaPipe landmark with .x, .y attributes)
        point_b: Second point (MediaPipe landmark with .x, .y attributes)
    
    Returns:
        distance: Normalized distance (0.0-1.0+)
    """
    a = np.array([point_a.x, point_a.y])
    b = np.array([point_b.x, point_b.y])
    return np.linalg.norm(a - b)

# Function to calculate angle between three points (in degrees)
def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle at point_b (vertex) formed by points a-b-c
    
    Args:
        point_a: First point (MediaPipe landmark with .x, .y attributes)
        point_b: Vertex point (the joint where angle is measured)
        point_c: Third point (MediaPipe landmark with .x, .y attributes)
    
    Returns:
        angle: Angle in degrees (0-180)
    """
    # Convert MediaPipe landmarks to numpy arrays
    a = np.array([point_a.x, point_a.y])
    b = np.array([point_b.x, point_b.y])
    c = np.array([point_c.x, point_c.y])
    
    # Calculate angles of vectors FROM vertex b TO points a and c
    angle_bc = np.arctan2(c[1] - b[1], c[0] - b[0])
    angle_ba = np.arctan2(a[1] - b[1], a[0] - b[0])
    
    # Difference between angles = angle between the two vectors
    radians = angle_bc - angle_ba
    
    # Convert radians to degrees and ensure positive value
    angle = np.abs(radians * 180.0 / np.pi)
        
    return angle

# Function to draw landmarks and connections
def draw_pose_landmarks(frame, landmarks, connections):
    """Draw pose landmarks and connections on the frame"""
    h, w, _ = frame.shape
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    # Draw landmarks
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

# Function to draw angle text on frame
def draw_angles(frame, landmarks, y_start=30, line_height=25):
    """
    Calculate and draw all joint angles on the frame
    
    Returns:
        angles_dict: Dictionary of calculated angles
    """
    h, w, _ = frame.shape
    y_offset = y_start
    angles_dict = {}
    
    # Left elbow angle (shoulder-elbow-wrist)
    if (LANDMARKS['LEFT_SHOULDER'] < len(landmarks) and 
        LANDMARKS['LEFT_ELBOW'] < len(landmarks) and 
        LANDMARKS['LEFT_WRIST'] < len(landmarks)):
        left_elbow_angle = calculate_angle(
            landmarks[LANDMARKS['LEFT_SHOULDER']],
            landmarks[LANDMARKS['LEFT_ELBOW']],
            landmarks[LANDMARKS['LEFT_WRIST']]
        )
        cv2.putText(frame, f'Left Elbow: {int(left_elbow_angle)}°', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        angles_dict['left_elbow'] = left_elbow_angle
        y_offset += line_height
    
    # Right elbow angle
    if (LANDMARKS['RIGHT_SHOULDER'] < len(landmarks) and 
        LANDMARKS['RIGHT_ELBOW'] < len(landmarks) and 
        LANDMARKS['RIGHT_WRIST'] < len(landmarks)):
        right_elbow_angle = calculate_angle(
            landmarks[LANDMARKS['RIGHT_SHOULDER']],
            landmarks[LANDMARKS['RIGHT_ELBOW']],
            landmarks[LANDMARKS['RIGHT_WRIST']]
        )
        cv2.putText(frame, f'Right Elbow: {int(right_elbow_angle)}°', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        angles_dict['right_elbow'] = right_elbow_angle
        y_offset += line_height
    
    # Left knee angle (hip-knee-ankle)
    if (LANDMARKS['LEFT_HIP'] < len(landmarks) and 
        LANDMARKS['LEFT_KNEE'] < len(landmarks) and 
        LANDMARKS['LEFT_ANKLE'] < len(landmarks)):
        left_knee_angle = calculate_angle(
            landmarks[LANDMARKS['LEFT_HIP']],
            landmarks[LANDMARKS['LEFT_KNEE']],
            landmarks[LANDMARKS['LEFT_ANKLE']]
        )
        cv2.putText(frame, f'Left Knee: {int(left_knee_angle)}°', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        angles_dict['left_knee'] = left_knee_angle
        y_offset += line_height
    
    # Right knee angle
    if (LANDMARKS['RIGHT_HIP'] < len(landmarks) and 
        LANDMARKS['RIGHT_KNEE'] < len(landmarks) and 
        LANDMARKS['RIGHT_ANKLE'] < len(landmarks)):
        right_knee_angle = calculate_angle(
            landmarks[LANDMARKS['RIGHT_HIP']],
            landmarks[LANDMARKS['RIGHT_KNEE']],
            landmarks[LANDMARKS['RIGHT_ANKLE']]
        )
        cv2.putText(frame, f'Right Knee: {int(right_knee_angle)}°', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        angles_dict['right_knee'] = right_knee_angle
        y_offset += line_height
    
    # Left shoulder angle (elbow-shoulder-hip)
    if (LANDMARKS['LEFT_ELBOW'] < len(landmarks) and 
        LANDMARKS['LEFT_SHOULDER'] < len(landmarks) and 
        LANDMARKS['LEFT_HIP'] < len(landmarks)):
        left_shoulder_angle = calculate_angle(
            landmarks[LANDMARKS['LEFT_ELBOW']],
            landmarks[LANDMARKS['LEFT_SHOULDER']],
            landmarks[LANDMARKS['LEFT_HIP']]
        )
        cv2.putText(frame, f'Left Shoulder: {int(left_shoulder_angle)}°', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        angles_dict['left_shoulder'] = left_shoulder_angle
        y_offset += line_height
    
    # Right shoulder angle
    if (LANDMARKS['RIGHT_ELBOW'] < len(landmarks) and 
        LANDMARKS['RIGHT_SHOULDER'] < len(landmarks) and 
        LANDMARKS['RIGHT_HIP'] < len(landmarks)):
        right_shoulder_angle = calculate_angle(
            landmarks[LANDMARKS['RIGHT_ELBOW']],
            landmarks[LANDMARKS['RIGHT_SHOULDER']],
            landmarks[LANDMARKS['RIGHT_HIP']]
        )
        cv2.putText(frame, f'Right Shoulder: {int(right_shoulder_angle)}°', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        angles_dict['right_shoulder'] = right_shoulder_angle
    
    return angles_dict

# ============================================================================
# GYMNASTICS JUDGING MEASUREMENTS
# ============================================================================

def measure_gymnastics_criteria(landmarks, frame_height, frame_width):
    """
    Measure three gymnastics criteria:
    1. Leg split angle (180° vertical split)
    2. Hands/arms balance (not touching ground)
    3. Releve (tip-toed feet position)
    
    Args:
        landmarks: List of MediaPipe pose landmarks
        frame_height: Height of frame in pixels
        frame_width: Width of frame in pixels
    
    Returns:
        dict: Structured data with measurements for each criterion
    """
    measurements = {
        'leg_split': {
            'angle_between_legs': None,
            'left_leg_straightness': None,  # Knee angle (180° = straight)
            'right_leg_straightness': None,
            'left_ankle_y': None,  # Normalized y position (lower = higher value)
            'right_ankle_y': None,
            'lower_foot': None,  # 'left' or 'right'
            'deviation_from_180': None  # How far from perfect 180°
        },
        'balance': {
            'left_wrist_y': None,  # Normalized y position
            'right_wrist_y': None,
            'lowest_ankle_y': None,  # Reference point for ground
            'left_hand_touching_ground': None,  # Boolean
            'right_hand_touching_ground': None,
            'any_hand_touching_ground': None
        },
        'releve': {
            'left_leg_vertical_angle': None,  # Angle of leg from vertical (0° = perfect)
            'right_leg_vertical_angle': None,
            'left_knee_angle': None,  # Should be ~180° for straight leg
            'right_knee_angle': None,
            'left_ankle_above_threshold': None,  # Boolean - heel raised
            'right_ankle_above_threshold': None
        }
    }
    
    # Check if required landmarks exist
    required = ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 
                'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_WRIST', 'RIGHT_WRIST']
    
    has_all = all(LANDMARKS[key] < len(landmarks) for key in required)
    
    if not has_all:
        return measurements
    
    # Get landmark objects
    left_hip = landmarks[LANDMARKS['LEFT_HIP']]
    right_hip = landmarks[LANDMARKS['RIGHT_HIP']]
    left_knee = landmarks[LANDMARKS['LEFT_KNEE']]
    right_knee = landmarks[LANDMARKS['RIGHT_KNEE']]
    left_ankle = landmarks[LANDMARKS['LEFT_ANKLE']]
    right_ankle = landmarks[LANDMARKS['RIGHT_ANKLE']]
    left_wrist = landmarks[LANDMARKS['LEFT_WRIST']]
    right_wrist = landmarks[LANDMARKS['RIGHT_WRIST']]
    
    # Calculate Mid-Hip Point (Virtual Landmark)
    # This creates a more stable center point for measuring the split
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            
    mid_hip = Point(
        (left_hip.x + right_hip.x) / 2,
        (left_hip.y + right_hip.y) / 2
    )
    
    # ========================================================================
    # CRITERION 1: LEG SPLIT AT 180° (Vertical Split)
    # ========================================================================
    
    # Measure knee angles (straightness of legs)
    # For a perfect split, both legs should be straight (knee angle ~180°)
    if (LANDMARKS['LEFT_HIP'] < len(landmarks) and 
        LANDMARKS['LEFT_KNEE'] < len(landmarks) and 
        LANDMARKS['LEFT_ANKLE'] < len(landmarks)):
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        measurements['leg_split']['left_leg_straightness'] = left_knee_angle
        measurements['releve']['left_knee_angle'] = left_knee_angle
    
    if (LANDMARKS['RIGHT_HIP'] < len(landmarks) and 
        LANDMARKS['RIGHT_KNEE'] < len(landmarks) and 
        LANDMARKS['RIGHT_ANKLE'] < len(landmarks)):
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        measurements['leg_split']['right_leg_straightness'] = right_knee_angle
        measurements['releve']['right_knee_angle'] = right_knee_angle
    
    # Measure angle between the two legs relative to the MID-HIP center
    # This is more precise than using one hip as the vertex
    # We calculate angle: left_ankle - mid_hip - right_ankle
    angle_between_legs = calculate_angle(left_ankle, mid_hip, right_ankle)
    
    measurements['leg_split']['angle_between_legs'] = angle_between_legs
    measurements['leg_split']['deviation_from_180'] = abs(180.0 - angle_between_legs)
    
    # Add mid-hip coordinates to measurements for visualization
    measurements['leg_split']['mid_hip_x'] = mid_hip.x
    measurements['leg_split']['mid_hip_y'] = mid_hip.y
    
    # Determine which foot is lower (closer to ground)
    # In normalized coordinates, higher y = lower on screen
    measurements['leg_split']['left_ankle_y'] = left_ankle.y
    measurements['leg_split']['right_ankle_y'] = right_ankle.y
    
    if left_ankle.y > right_ankle.y:  # Left ankle is lower
        measurements['leg_split']['lower_foot'] = 'left'
    else:
        measurements['leg_split']['lower_foot'] = 'right'
    
    # ========================================================================
    # CRITERION 2: BALANCE - HANDS/ARMS NOT TOUCHING GROUND
    # ========================================================================
    
    # Get wrist positions
    measurements['balance']['left_wrist_y'] = left_wrist.y
    measurements['balance']['right_wrist_y'] = right_wrist.y
    
    # Use the lower ankle as reference for "ground level"
    lowest_ankle_y = max(left_ankle.y, right_ankle.y)
    measurements['balance']['lowest_ankle_y'] = lowest_ankle_y
    
    # Define threshold: if wrist is below ankle level (with some tolerance), it's touching
    # Tolerance: 0.02 (2% of frame height) to account for perspective
    ground_threshold = lowest_ankle_y + 0.02
    
    left_touching = left_wrist.y > ground_threshold
    right_touching = right_wrist.y > ground_threshold
    
    measurements['balance']['left_hand_touching_ground'] = left_touching
    measurements['balance']['right_hand_touching_ground'] = right_touching
    measurements['balance']['any_hand_touching_ground'] = left_touching or right_touching
    
    # ========================================================================
    # CRITERION 3: RELEVE (Tip-toed feet, not flat)
    # ========================================================================
    
    # For releve, the leg should be straight and vertical
    # Measure the angle of each leg from vertical (0° = perfectly vertical)
    
    # Left leg vertical angle
    # Calculate angle from vertical: hip to ankle should be vertical
    # We measure deviation from vertical by comparing to a vertical line
    left_hip_ankle_angle = np.arctan2(
        abs(left_ankle.x - left_hip.x),  # Horizontal distance
        abs(left_ankle.y - left_hip.y)   # Vertical distance
    ) * 180.0 / np.pi
    measurements['releve']['left_leg_vertical_angle'] = left_hip_ankle_angle
    
    # Right leg vertical angle
    right_hip_ankle_angle = np.arctan2(
        abs(right_ankle.x - right_hip.x),
        abs(right_ankle.y - right_hip.y)
    ) * 180.0 / np.pi
    measurements['releve']['right_leg_vertical_angle'] = right_hip_ankle_angle
    
    # Check if ankle is "raised" (for releve, the heel should be off ground)
    # In a releve, the ankle should be relatively high compared to a flat foot
    # We compare ankle position to knee position - in releve, ankle should be higher
    # (lower y value in normalized coordinates)
    
    # For a proper releve, the leg should be straight (knee angle ~180°)
    # and the ankle should be at a similar or higher level than knee
    # (accounting for the fact that in releve, you're on your toes)
    
    # Simple check: if knee angle is close to 180° (straight leg), it's likely releve
    # More precise: compare ankle y to a reference point
    # In releve, the foot is extended, so ankle might be slightly higher
    
    # Use knee as reference - in releve, ankle should be at similar or higher level
    left_ankle_raised = left_ankle.y <= left_knee.y + 0.01  # Small tolerance
    right_ankle_raised = right_ankle.y <= right_knee.y + 0.01
    
    measurements['releve']['left_ankle_above_threshold'] = left_ankle_raised
    measurements['releve']['right_ankle_above_threshold'] = right_ankle_raised
    
    return measurements

def format_measurements_for_api(measurements):
    """
    Format measurements into a clean structure ready for AI API
    
    Returns:
        dict: Formatted data with key metrics
    """
    formatted = {
        'leg_split_180': {
            'angle_between_legs_degrees': measurements['leg_split']['angle_between_legs'],
            'deviation_from_180_degrees': measurements['leg_split']['deviation_from_180'],
            'left_leg_straightness_degrees': measurements['leg_split']['left_leg_straightness'],
            'right_leg_straightness_degrees': measurements['leg_split']['right_leg_straightness'],
            'lower_foot': measurements['leg_split']['lower_foot'],
            'is_perfect_180': measurements['leg_split']['deviation_from_180'] is not None and 
                             measurements['leg_split']['deviation_from_180'] < 5.0  # Within 5° tolerance
        },
        'balance_no_hands_ground': {
            'left_wrist_y_normalized': measurements['balance']['left_wrist_y'],
            'right_wrist_y_normalized': measurements['balance']['right_wrist_y'],
            'ground_reference_y_normalized': measurements['balance']['lowest_ankle_y'],
            'left_hand_touching': measurements['balance']['left_hand_touching_ground'],
            'right_hand_touching': measurements['balance']['right_hand_touching_ground'],
            'any_hand_touching': measurements['balance']['any_hand_touching_ground'],
            'balance_maintained': not measurements['balance']['any_hand_touching_ground'] if measurements['balance']['any_hand_touching_ground'] is not None else None
        },
        'releve_tip_toed': {
            'left_leg_vertical_angle_degrees': measurements['releve']['left_leg_vertical_angle'],
            'right_leg_vertical_angle_degrees': measurements['releve']['right_leg_vertical_angle'],
            'left_knee_angle_degrees': measurements['releve']['left_knee_angle'],
            'right_knee_angle_degrees': measurements['releve']['right_knee_angle'],
            'left_ankle_raised': measurements['releve']['left_ankle_above_threshold'],
            'right_ankle_raised': measurements['releve']['right_ankle_above_threshold'],
            'both_feet_releve': (measurements['releve']['left_ankle_above_threshold'] and 
                               measurements['releve']['right_ankle_above_threshold']) if (
                               measurements['releve']['left_ankle_above_threshold'] is not None and
                               measurements['releve']['right_ankle_above_threshold'] is not None) else None
        }
    }
    
    return formatted

# Initialize MediaPipe Pose Landmarker
MODEL_PATHS = ["pose_landmarker_full.task", "pose_landmarker_lite.task"]
MODEL_PATH = None

for path in MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    print("Model file not found!")
    print("Please download one of the following:")
    print("Full model (more accurate):")
    print("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
    print("\nLite model (faster):")
    print("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
    sys.exit(1)

print(f"Using model: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# Pose landmark connections for drawing
POSE_CONNECTIONS = [
    # Face oval
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Torso
    (9, 10),   # Left shoulder to right shoulder
    (11, 12),  # Left hip to right hip
    (11, 9),   # Left shoulder to left hip
    (12, 10),  # Right shoulder to right hip
    # Left arm
    (11, 13),  # Left shoulder to left elbow
    (13, 15),  # Left elbow to left wrist
    # Right arm
    (12, 14),  # Right shoulder to right elbow
    (14, 16),  # Right elbow to right wrist
    # Left leg
    (23, 25),  # Left hip to left knee
    (25, 27),  # Left knee to left ankle
    # Right leg
    (24, 26),  # Right hip to right knee
    (26, 28),  # Right knee to right ankle
]

# Pose landmark indices
LANDMARKS = {
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
}

def analyze_video(video_path, output_path=None, save_output=False):
    """
    Analyze a video file and perform pose detection with angle calculations
    
    Args:
        video_path: Path to input video file (.mp4)
        output_path: Path to save output video (optional)
        save_output: Whether to save the analyzed video
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo Information:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"\nProcessing video...")
    print("Press 'q' to quit, SPACE to pause/resume")
    
    # Setup video writer if saving output
    out = None
    if save_output:
        if output_path is None:
            # Auto-generate output filename
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}_analyzed.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"  Saving output to: {output_path}")
    
    frame_count = 0
    paused = False
    all_angles = []  # Store angles for all frames
    all_gymnastics_measurements = []  # Store gymnastics criteria measurements
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect poses
            detection_result = detector.detect(mp_image)
            
            frame_angles = {}
            
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                
                # Draw pose landmarks and connections
                draw_pose_landmarks(frame, landmarks, POSE_CONNECTIONS)
                
                # Calculate and draw angles
                frame_angles = draw_angles(frame, landmarks)
                
                # Measure gymnastics criteria
                gymnastics_measurements = measure_gymnastics_criteria(
                    landmarks, height, width
                )
                
                # Format for API
                formatted_measurements = format_measurements_for_api(gymnastics_measurements)
                
                # Store measurements
                all_gymnastics_measurements.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'measurements': formatted_measurements
                })
                
                # Draw gymnastics criteria on frame
                y_pos = height - 100
                if gymnastics_measurements['leg_split']['angle_between_legs'] is not None:
                    split_angle = gymnastics_measurements['leg_split']['angle_between_legs']
                    deviation = gymnastics_measurements['leg_split']['deviation_from_180']
                    
                    # Draw mid-hip point and lines to ankles for visualization
                    if 'mid_hip_x' in gymnastics_measurements['leg_split']:
                        mx = int(gymnastics_measurements['leg_split']['mid_hip_x'] * width)
                        my = int(gymnastics_measurements['leg_split']['mid_hip_y'] * height)
                        lax = int(landmarks[LANDMARKS['LEFT_ANKLE']].x * width)
                        lay = int(landmarks[LANDMARKS['LEFT_ANKLE']].y * height)
                        rax = int(landmarks[LANDMARKS['RIGHT_ANKLE']].x * width)
                        ray = int(landmarks[LANDMARKS['RIGHT_ANKLE']].y * height)
                        
                        # Draw center point (cyan)
                        cv2.circle(frame, (mx, my), 6, (255, 255, 0), -1)
                        # Draw lines from center to ankles (cyan)
                        cv2.line(frame, (mx, my), (lax, lay), (255, 255, 0), 2)
                        cv2.line(frame, (mx, my), (rax, ray), (255, 255, 0), 2)

                    cv2.putText(frame, f'Split: {split_angle:.1f}° (dev: {deviation:.1f}°)', 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_pos -= 25
                
                if gymnastics_measurements['balance']['any_hand_touching_ground'] is not None:
                    balance_status = "BALANCED" if not gymnastics_measurements['balance']['any_hand_touching_ground'] else "HANDS DOWN"
                    color = (0, 255, 0) if not gymnastics_measurements['balance']['any_hand_touching_ground'] else (0, 0, 255)
                    cv2.putText(frame, f'Balance: {balance_status}', 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos -= 25
                
                if (gymnastics_measurements['releve']['left_ankle_above_threshold'] is not None and
                    gymnastics_measurements['releve']['right_ankle_above_threshold'] is not None):
                    both_releve = (gymnastics_measurements['releve']['left_ankle_above_threshold'] and
                                 gymnastics_measurements['releve']['right_ankle_above_threshold'])
                    releve_status = "RELEVE" if both_releve else "FLAT FOOT"
                    color = (0, 255, 0) if both_releve else (0, 0, 255)
                    cv2.putText(frame, f'Feet: {releve_status}', 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add frame info overlay
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
            cv2.putText(frame, info_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Store angles for this frame
            all_angles.append({
                'frame': frame_count,
                'angles': frame_angles if 'frame_angles' in locals() else {}
            })
            
            # Save frame if output enabled
            if save_output and out is not None:
                out.write(frame)
            
            # Display the frame
            cv2.imshow('Video Analysis - MediaPipe Pose Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar to pause/resume
            paused = not paused
            print(f"  {'Paused' if paused else 'Resumed'}")
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\nAnalysis complete!")
    print(f"  Processed {frame_count} frames")
    if save_output:
        print(f"  Output saved to: {output_path}")
    
    # Print angle statistics
    if all_angles:
        print(f"\nAngle Statistics:")
        angle_names = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 
                      'left_shoulder', 'right_shoulder']
        
        for angle_name in angle_names:
            angles = [f['angles'].get(angle_name) for f in all_angles 
                     if angle_name in f['angles']]
            if angles:
                avg_angle = np.mean(angles)
                min_angle = np.min(angles)
                max_angle = np.max(angles)
                print(f"  {angle_name.replace('_', ' ').title()}: "
                      f"Avg={avg_angle:.1f}°, Min={min_angle:.1f}°, Max={max_angle:.1f}°")
    
    # Print gymnastics criteria summary
    if all_gymnastics_measurements:
        print(f"\n" + "=" * 60)
        print("GYMNASTICS CRITERIA SUMMARY")
        print("=" * 60)
        
        # Leg split statistics
        split_angles = [m['measurements']['leg_split_180']['angle_between_legs_degrees'] 
                       for m in all_gymnastics_measurements 
                       if m['measurements']['leg_split_180']['angle_between_legs_degrees'] is not None]
        if split_angles:
            avg_split = np.mean(split_angles)
            min_split = np.min(split_angles)
            max_split = np.max(split_angles)
            avg_deviation = np.mean([m['measurements']['leg_split_180']['deviation_from_180_degrees'] 
                                    for m in all_gymnastics_measurements 
                                    if m['measurements']['leg_split_180']['deviation_from_180_degrees'] is not None])
            print(f"\n1. LEG SPLIT (180° Target):")
            print(f"   Average angle: {avg_split:.1f}°")
            print(f"   Range: {min_split:.1f}° - {max_split:.1f}°")
            print(f"   Average deviation from 180°: {avg_deviation:.1f}°")
        
        # Balance statistics
        hands_touching_frames = sum([1 for m in all_gymnastics_measurements 
                                    if m['measurements']['balance_no_hands_ground']['any_hand_touching']])
        total_frames_with_balance = len([m for m in all_gymnastics_measurements 
                                        if m['measurements']['balance_no_hands_ground']['any_hand_touching'] is not None])
        if total_frames_with_balance > 0:
            balance_percentage = (1 - hands_touching_frames / total_frames_with_balance) * 100
            print(f"\n2. BALANCE (No Hands on Ground):")
            print(f"   Frames with hands touching ground: {hands_touching_frames}/{total_frames_with_balance}")
            print(f"   Balance maintained: {balance_percentage:.1f}% of frames")
        
        # Releve statistics
        releve_frames = sum([1 for m in all_gymnastics_measurements 
                           if m['measurements']['releve_tip_toed']['both_feet_releve']])
        total_frames_with_releve = len([m for m in all_gymnastics_measurements 
                                       if m['measurements']['releve_tip_toed']['both_feet_releve'] is not None])
        if total_frames_with_releve > 0:
            releve_percentage = (releve_frames / total_frames_with_releve) * 100
            print(f"\n3. RELEVE (Tip-toed Feet):")
            print(f"   Frames in releve: {releve_frames}/{total_frames_with_releve}")
            print(f"   Releve maintained: {releve_percentage:.1f}% of frames")
        
        # Return formatted data for API
        # Find the frame with maximum leg split (peak of the move)
        peak_frame = None
        max_split_angle = -1
        peak_frame_index = -1
        
        for i, m in enumerate(all_gymnastics_measurements):
            angle = m['measurements']['leg_split_180']['angle_between_legs_degrees']
            if angle is not None and angle > max_split_angle:
                max_split_angle = angle
                peak_frame = m
                peak_frame_index = i
        
        # Calculate 1-second window stats (0.5s before to 0.5s after peak)
        hold_window_stats = None
        if peak_frame:
            # Calculate frame range for 1 second (fps frames)
            half_window = int(fps / 2)
            start_idx = max(0, peak_frame_index - half_window)
            end_idx = min(len(all_gymnastics_measurements), peak_frame_index + half_window + 1)
            
            window_frames = all_gymnastics_measurements[start_idx:end_idx]
            
            # 1. Balance check (Must NOT touch ground at all during window)
            hands_touched = any(f['measurements']['balance_no_hands_ground']['any_hand_touching'] for f in window_frames)
            
            # 2. Min split angle (The lowest angle during the "hold")
            window_angles = [f['measurements']['leg_split_180']['angle_between_legs_degrees'] for f in window_frames 
                           if f['measurements']['leg_split_180']['angle_between_legs_degrees'] is not None]
            min_hold_angle = min(window_angles) if window_angles else 0
            
            # 3. Releve check (Maintained for >50% of window?)
            releve_frames_count = sum(1 for f in window_frames if f['measurements']['releve_tip_toed']['both_feet_releve'])
            releve_maintained = releve_frames_count >= (len(window_frames) / 2)
            
            hold_window_stats = {
                'window_start_frame': window_frames[0]['frame'],
                'window_end_frame': window_frames[-1]['frame'],
                'duration_frames': len(window_frames),
                'min_split_angle_during_hold': float(min_hold_angle),
                'balance_maintained_throughout': not hands_touched,
                'releve_maintained_majority': releve_maintained
            }
                
        print(f"\nPEAK PERFORMANCE (Frame {peak_frame['frame'] if peak_frame else 'N/A'}):")
        if peak_frame:
            split = peak_frame['measurements']['leg_split_180']
            balance = peak_frame['measurements']['balance_no_hands_ground']
            releve = peak_frame['measurements']['releve_tip_toed']
            
            print(f"   Max Split Angle: {split['angle_between_legs_degrees']:.1f}°")
            
            if hold_window_stats:
                print(f"\n1-SECOND HOLD ANALYSIS (Frames {hold_window_stats['window_start_frame']}-{hold_window_stats['window_end_frame']}):")
                print(f"   Min Split During Hold: {hold_window_stats['min_split_angle_during_hold']:.1f}°")
                print(f"   Balance Maintained (1s): {'Yes' if hold_window_stats['balance_maintained_throughout'] else 'NO - Hands touched ground'}")
                print(f"   Releve Maintained (1s): {'Yes' if hold_window_stats['releve_maintained_majority'] else 'No'}")

        return {
            'video_info': {
                'total_frames': total_frames,
                'fps': fps,
                'duration_seconds': duration,
                'resolution': f"{width}x{height}"
            },
            'peak_performance': peak_frame,
            'hold_window_1s': hold_window_stats,
            'summary_statistics': {
                'max_split_angle': float(max_split_angle) if max_split_angle > 0 else None,
                'overall_balance_percentage': float(balance_percentage) if total_frames_with_balance > 0 else None
            }
        }
    
    return None

def export_measurements_to_json(measurements_data, output_file):
    """
    Export measurements data to JSON file for AI API consumption
    
    Args:
        measurements_data: Dictionary returned from analyze_video()
        output_file: Path to output JSON file
    """
    if measurements_data is None:
        print("No measurements data to export")
        return
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_data = convert_to_serializable(measurements_data)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"\nMeasurements exported to: {output_file}")

def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Analyze video files with MediaPipe pose detection and angle measurement'
    )
    parser.add_argument('video_path', type=str, 
                       help='Path to input video file (.mp4)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Path to save output video (default: auto-generated)')
    parser.add_argument('-s', '--save', action='store_true',
                       help='Save analyzed video to file')
    parser.add_argument('-j', '--json', type=str, default=None,
                       help='Export measurements to JSON file')
    
    args = parser.parse_args()
    
    result = analyze_video(args.video_path, args.output, args.save)
    
    # Export to JSON if requested
    if args.json and result:
        export_measurements_to_json(result, args.json)

if __name__ == "__main__":
    # If run without arguments, prompt for video file
    if len(sys.argv) == 1:
        print("Video Analyzer - MediaPipe Pose Detection")
        print("=" * 50)
        
        # Look for videos directory
        videos_dir = "videos"
        if not os.path.exists(videos_dir):
            videos_dir = "."  # Fallback to current directory if 'videos' doesn't exist
            
        # List all mp4 files
        video_files = []
        for root, dirs, files in os.walk(videos_dir):
            for file in files:
                if file.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
        
        video_path = None
        
        if video_files:
            print(f"\nFound {len(video_files)} videos in '{videos_dir}':")
            for i, vid in enumerate(video_files):
                print(f"  {i+1}. {os.path.basename(vid)}")
            print(f"  0. Enter manual path")
            
            try:
                choice = input("\nSelect video (0-{0}): ".format(len(video_files))).strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 0 < idx <= len(video_files):
                        video_path = video_files[idx-1]
            except ValueError:
                pass
        
        if not video_path:
            video_path = input("\nEnter path to video file (.mp4): ").strip().strip('"')
        
        if not video_path:
            print("No video path provided. Exiting.")
            sys.exit(1)
            
        print(f"\nSelected: {video_path}")
        
        save_choice = input("Save output video? (y/n): ").strip().lower()
        save_output = save_choice in ['y', 'yes']
        
        output_path = None
        if save_output:
            output_path = input("Output path (press Enter for auto-generated): ").strip().strip('"')
            if not output_path:
                output_path = None
        
        result = analyze_video(video_path, output_path, save_output)
        
        # Ask if user wants to export to JSON
        export_choice = input("\nExport measurements to JSON? (y/n): ").strip().lower()
        if export_choice in ['y', 'yes']:
            json_path = input("JSON output path (press Enter for auto-generated): ").strip().strip('"')
            if not json_path:
                base_name = os.path.splitext(video_path)[0]
                json_path = f"{base_name}_measurements.json"
            export_measurements_to_json(result, json_path)
    else:
        main()
