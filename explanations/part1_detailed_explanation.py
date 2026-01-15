"""
DETAILED EXPLANATION: Part 1 - Drawing Function
Breaking down every line with examples and visualizations
"""

import cv2
import numpy as np

# ============================================================================
# UNDERSTANDING THE INPUTS
# ============================================================================

# Example: What does MediaPipe give us?
class ExampleLandmark:
    """This is what MediaPipe landmark objects look like"""
    def __init__(self, x, y):
        self.x = x  # Normalized coordinate: 0.0 to 1.0
        self.y = y  # Normalized coordinate: 0.0 to 1.0
        # MediaPipe also has .z and .visibility, but we only use .x and .y

# Example landmarks (pretend these came from MediaPipe)
example_landmarks = [
    ExampleLandmark(0.5, 0.2),  # Landmark 0: center-top of face
    ExampleLandmark(0.4, 0.3),  # Landmark 1: left eye
    ExampleLandmark(0.6, 0.3),  # Landmark 2: right eye
    ExampleLandmark(0.5, 0.5),  # Landmark 11: left shoulder
    ExampleLandmark(0.5, 0.7),  # Landmark 13: left elbow
    ExampleLandmark(0.5, 0.9),  # Landmark 15: left wrist
]

# Example connections (which landmarks to connect with lines)
example_connections = [
    (0, 1),   # Connect landmark 0 to landmark 1
    (1, 2),   # Connect landmark 1 to landmark 2
    (11, 13), # Connect left shoulder to left elbow
    (13, 15), # Connect left elbow to left wrist
]

# Example frame (image from camera)
# This is what OpenCV gives us - a NumPy array
example_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image: 480px tall, 640px wide, 3 color channels

print("=" * 60)
print("EXAMPLE VALUES:")
print("=" * 60)
print(f"Frame shape: {example_frame.shape}")  # (480, 640, 3)
print(f"Frame height: {example_frame.shape[0]}")  # 480
print(f"Frame width: {example_frame.shape[1]}")  # 640
print(f"Frame channels: {example_frame.shape[2]}")  # 3 (BGR)
print(f"\nExample landmark 0: x={example_landmarks[0].x}, y={example_landmarks[0].y}")
print(f"Example landmark 11 (shoulder): x={example_landmarks[3].x}, y={example_landmarks[3].y}")

# ============================================================================
# LINE-BY-LINE BREAKDOWN OF draw_pose_landmarks()
# ============================================================================

def draw_pose_landmarks_explained(frame, landmarks, connections):
    """
    Draw pose landmarks and connections on the frame
    
    PARAMETERS:
    - frame: NumPy array representing the image (from camera)
    - landmarks: List of MediaPipe landmark objects (each has .x, .y)
    - connections: List of tuples like [(0,1), (1,2), ...] showing which landmarks to connect
    """
    
    # ========================================================================
    # LINE 66: h, w, _ = frame.shape
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Getting frame dimensions")
    print("=" * 60)
    
    # frame.shape returns a tuple: (height, width, channels)
    shape_tuple = frame.shape
    print(f"frame.shape = {shape_tuple}")  # Example: (480, 640, 3)
    
    # Unpacking the tuple into variables
    h = frame.shape[0]  # Height (number of rows/pixels vertically)
    w = frame.shape[1]  # Width (number of columns/pixels horizontally)
    _ = frame.shape[2]  # Channels (we ignore this with _)
    
    # Python shortcut: h, w, _ = frame.shape does all three at once!
    h, w, _ = frame.shape
    
    print(f"h (height) = {h} pixels")
    print(f"w (width) = {w} pixels")
    print(f"_ (channels, ignored) = {frame.shape[2]}")
    
    # ========================================================================
    # LINES 69-79: Drawing connections (lines between joints)
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Drawing connections (lines)")
    print("=" * 60)
    
    for connection in connections:
        print(f"\nProcessing connection: {connection}")
        
        # ====================================================================
        # LINE 70: start_idx, end_idx = connection
        # ====================================================================
        # connection is a tuple like (11, 13)
        # Tuple unpacking: assigns first value to start_idx, second to end_idx
        start_idx, end_idx = connection
        print(f"  start_idx = {start_idx}, end_idx = {end_idx}")
        
        # ====================================================================
        # LINE 72: if start_idx < len(landmarks) and end_idx < len(landmarks):
        # ====================================================================
        # Safety check: Make sure these indices actually exist in the landmarks list
        num_landmarks = len(landmarks)
        print(f"  Number of landmarks available: {num_landmarks}")
        
        if start_idx < num_landmarks and end_idx < num_landmarks:
            print(f"  ✓ Both indices are valid!")
            
            # ================================================================
            # LINES 73-74: Get the actual landmark objects
            # ================================================================
            start = landmarks[start_idx]  # Get landmark at index start_idx
            end = landmarks[end_idx]      # Get landmark at index end_idx
            
            print(f"  start landmark: x={start.x}, y={start.y}")
            print(f"  end landmark: x={end.x}, y={end.y}")
            
            # ================================================================
            # LINES 76-77: Convert normalized coordinates to pixel coordinates
            # ================================================================
            # MediaPipe gives us coordinates from 0.0 to 1.0 (normalized)
            # OpenCV needs actual pixel positions (0 to width/height)
            
            # Example calculation:
            # If frame width = 640 and landmark.x = 0.5:
            # pixel_x = 0.5 * 640 = 320 pixels from left edge
            
            start_point_x = int(start.x * w)  # Convert normalized x to pixel x
            start_point_y = int(start.y * h)  # Convert normalized y to pixel y
            start_point = (start_point_x, start_point_y)
            
            end_point_x = int(end.x * w)
            end_point_y = int(end.y * h)
            end_point = (end_point_x, end_point_y)
            
            print(f"  Converted to pixels:")
            print(f"    start_point = ({start_point_x}, {start_point_y})")
            print(f"    end_point = ({end_point_x}, {end_point_y})")
            
            # ================================================================
            # LINE 79: cv2.line() - Draw the line
            # ================================================================
            # cv2.line(image, start_point, end_point, color, thickness)
            # - image: The frame to draw on (modified in-place)
            # - start_point: (x, y) tuple for line start
            # - end_point: (x, y) tuple for line end
            # - color: (B, G, R) tuple - (0, 255, 0) = green
            # - thickness: Line width in pixels
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            print(f"  ✓ Drew green line from {start_point} to {end_point}")
        else:
            print(f"  ✗ Invalid indices! Skipping this connection.")
    
    # ========================================================================
    # LINES 82-85: Drawing landmarks (dots at each joint)
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Drawing landmarks (dots)")
    print("=" * 60)
    
    for i, landmark in enumerate(landmarks):
        print(f"\nLandmark {i}: x={landmark.x}, y={landmark.y}")
        
        # ====================================================================
        # LINES 83-84: Convert to pixel coordinates
        # ====================================================================
        x = int(landmark.x * w)  # Normalized x → pixel x
        y = int(landmark.y * h)  # Normalized y → pixel y
        
        print(f"  Pixel position: ({x}, {y})")
        
        # ====================================================================
        # LINE 85: cv2.circle() - Draw the dot
        # ====================================================================
        # cv2.circle(image, center, radius, color, thickness)
        # - image: The frame to draw on
        # - center: (x, y) tuple for circle center
        # - radius: Size of circle in pixels
        # - color: (B, G, R) tuple - (0, 0, 255) = red
        # - thickness: -1 means filled circle, positive = outline only
        
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        print(f"  ✓ Drew red circle at ({x}, {y})")

# ============================================================================
# VISUAL EXAMPLE WITH ACTUAL NUMBERS
# ============================================================================

print("\n" + "=" * 60)
print("CONCRETE EXAMPLE:")
print("=" * 60)

# Let's say we have a 640x480 camera frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image

# MediaPipe detected a person and gave us these landmarks:
landmarks = [
    ExampleLandmark(0.5, 0.2),   # Landmark 0: Top of head
    ExampleLandmark(0.4, 0.3),   # Landmark 1: Left eye
    ExampleLandmark(0.6, 0.3),   # Landmark 2: Right eye
    ExampleLandmark(0.5, 0.5),   # Landmark 11: Left shoulder
    ExampleLandmark(0.5, 0.7),   # Landmark 13: Left elbow
    ExampleLandmark(0.5, 0.9),   # Landmark 15: Left wrist
]

connections = [
    (0, 1),   # Connect head to left eye
    (11, 13), # Connect shoulder to elbow
    (13, 15), # Connect elbow to wrist
]

print("\nFrame dimensions: 640 pixels wide × 480 pixels tall")
print("\nLandmark coordinates (normalized 0.0-1.0):")
for i, lm in enumerate(landmarks):
    pixel_x = int(lm.x * 640)
    pixel_y = int(lm.y * 480)
    print(f"  Landmark {i}: ({lm.x}, {lm.y}) → Pixel ({pixel_x}, {pixel_y})")

print("\n" + "=" * 60)
print("Calling the function...")
print("=" * 60)

# Call the function with our example data
draw_pose_landmarks_explained(frame, landmarks, connections)

print("\n" + "=" * 60)
print("KEY CONCEPTS:")
print("=" * 60)
print("""
1. NORMALIZED COORDINATES (0.0 to 1.0):
   - MediaPipe gives coordinates as percentages
   - x=0.5 means "50% from left edge"
   - Works for any image size!

2. PIXEL COORDINATES (0 to width/height):
   - OpenCV needs actual pixel positions
   - Must convert: pixel = normalized × dimension
   - Example: 0.5 × 640 = 320 pixels

3. TUPLE UNPACKING:
   - (a, b) = (1, 2) assigns a=1, b=2
   - frame.shape returns (height, width, channels)
   - h, w, _ = frame.shape unpacks all three

4. IN-PLACE MODIFICATION:
   - cv2.line() and cv2.circle() modify the frame directly
   - No need to return or reassign
   - The frame array is changed permanently

5. BGR COLOR FORMAT:
   - OpenCV uses (Blue, Green, Red) not RGB!
   - (0, 255, 0) = Green
   - (0, 0, 255) = Red
   - (255, 255, 255) = White
""")
