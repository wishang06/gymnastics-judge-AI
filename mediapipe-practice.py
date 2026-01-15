import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os

# Function to calculate angle between three points (in degrees)
def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle at point_b (vertex) formed by points a-b-c
    
    Mathematical approach:
    1. Convert MediaPipe landmarks to 2D vectors [x, y]
    2. Calculate angle of vector b→c using arctan2
    3. Calculate angle of vector b→a using arctan2
    4. Difference gives the angle between them
    5. Convert radians to degrees
    
    Args:
        point_a: First point (MediaPipe landmark with .x, .y attributes)
        point_b: Vertex point (the joint where angle is measured)
        point_c: Third point (MediaPipe landmark with .x, .y attributes)
    
    Returns:
        angle: Angle in degrees (0-180)
    
    Example:
        For elbow angle: point_a=shoulder, point_b=elbow, point_c=wrist
    """
    # Step 1: Convert MediaPipe landmarks to numpy arrays
    # point_a.x, point_a.y are normalized coordinates (0.0 to 1.0)
    # np.array creates a 2D vector [x, y] for vector math
    a = np.array([point_a.x, point_a.y])  # [x_coordinate, y_coordinate]
    b = np.array([point_b.x, point_b.y])  # Vertex (middle point)
    c = np.array([point_c.x, point_c.y])  # Third point
    
    # Step 2: Calculate angles of vectors FROM vertex b TO points a and c
    # np.arctan2(y, x) returns angle in radians of vector (x, y)
    # It handles all 4 quadrants correctly (unlike np.arctan)
    
    # Angle of vector b→c: direction from elbow to wrist
    angle_bc = np.arctan2(c[1] - b[1], c[0] - b[0])
    #                    ↑ delta_y    ↑ delta_x
    # This asks: "What angle does line b→c make with horizontal?"
    
    # Angle of vector b→a: direction from elbow to shoulder  
    angle_ba = np.arctan2(a[1] - b[1], a[0] - b[0])
    #                    ↑ delta_y    ↑ delta_x
    # This asks: "What angle does line b→a make with horizontal?"
    
    # Step 3: Difference between angles = angle between the two vectors
    radians = angle_bc - angle_ba
    
    # Step 4: Convert radians to degrees and ensure positive value
    # 180 degrees = π radians, so: degrees = radians × (180 / π)
    angle = np.abs(radians * 180.0 / np.pi)
    # np.abs() ensures we get angle between 0-180 degrees
        
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

# Initialize MediaPipe Pose Landmarker
# Note: You need to download the model file from:
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
# Place it in the same directory as this script

# Try to find model file (full or lite version)
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
    print("\nOr run: python download_model.py")
    exit(1)

print(f"Using model: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# Pose landmark connections for drawing (MediaPipe Pose 33 landmarks)
# These are the standard MediaPipe pose connections
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

# Pose landmark indices (MediaPipe Pose landmarks)
# Key joints for angle measurement
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

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

print("Press 'q' to quit")
print("Measuring angles between joints...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect poses
    detection_result = detector.detect(mp_image)
    
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        
        # Draw pose landmarks and connections
        draw_pose_landmarks(frame, landmarks, POSE_CONNECTIONS)
        
        h, w, _ = frame.shape
        y_offset = 30
        line_height = 25
        
        # Calculate and display angles
        
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
    
    # Display the frame
    cv2.imshow('MediaPipe Pose - Joint Angle Measurement', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
