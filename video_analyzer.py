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
            
            # Add frame info overlay
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
            cv2.putText(frame, info_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Store angles for this frame
            all_angles.append({
                'frame': frame_count,
                'angles': frame_angles
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
    
    args = parser.parse_args()
    
    analyze_video(args.video_path, args.output, args.save)

if __name__ == "__main__":
    # If run without arguments, prompt for video file
    if len(sys.argv) == 1:
        print("Video Analyzer - MediaPipe Pose Detection")
        print("=" * 50)
        video_path = input("Enter path to video file (.mp4): ").strip().strip('"')
        
        if not video_path:
            print("No video path provided. Exiting.")
            sys.exit(1)
        
        save_choice = input("Save output video? (y/n): ").strip().lower()
        save_output = save_choice in ['y', 'yes']
        
        output_path = None
        if save_output:
            output_path = input("Output path (press Enter for auto-generated): ").strip().strip('"')
            if not output_path:
                output_path = None
        
        analyze_video(video_path, output_path, save_output)
    else:
        main()
