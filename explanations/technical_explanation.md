# Technical Explanation: MediaPipe Pose Detection & Angle Measurement

## Part 1: Drawing Function (Lines 63-85)

### `draw_pose_landmarks()` Function

```python
def draw_pose_landmarks(frame, landmarks, connections):
    h, w, _ = frame.shape
```

**Line 66: `h, w, _ = frame.shape`**
- `frame.shape` returns a tuple: `(height, width, channels)`
- `h, w, _` unpacks: height, width, and ignores channels (the `_`)
- Example: `(480, 640, 3)` → `h=480, w=640`

**Lines 69-79: Drawing Connections**
```python
for connection in connections:
    start_idx, end_idx = connection
```
- `connections` is a list of tuples: `[(0, 1), (1, 2), ...]`
- **Tuple unpacking**: `start_idx, end_idx = (0, 1)` assigns `start_idx=0, end_idx=1`
- Each tuple represents which landmarks to connect

```python
if start_idx < len(landmarks) and end_idx < len(landmarks):
```
- **Bounds checking**: Ensures indices exist before accessing
- Prevents `IndexError` if model returns fewer landmarks

```python
start_point = (int(start.x * w), int(start.y * h))
end_point = (int(end.x * w), int(end.y * h))
```
- **Coordinate conversion**: MediaPipe gives normalized coordinates (0.0-1.0)
- `start.x * w`: Converts 0.0-1.0 to pixel coordinates (0 to width)
- `int()`: OpenCV needs integer pixel coordinates
- Example: If `w=640` and `start.x=0.5`, then `x = 320` pixels

```python
cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
```
- **OpenCV line drawing**: `cv2.line(image, start, end, color, thickness)`
- `(0, 255, 0)`: BGR color format (green in this case)
- `2`: Line thickness in pixels

**Lines 82-85: Drawing Landmarks**
```python
for landmark in landmarks:
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
```
- **Circle drawing**: `cv2.circle(image, center, radius, color, thickness)`
- `5`: Radius in pixels
- `(0, 0, 255)`: Red in BGR format
- `-1`: Filled circle (negative = fill, positive = outline)

---

## Part 2: Model Initialization (Lines 87-118)

### Model File Detection (Lines 92-111)

```python
MODEL_PATHS = ["pose_landmarker_full.task", "pose_landmarker_lite.task"]
MODEL_PATH = None
```
- **List of possible model files**: Checks both full and lite versions
- `None`: Initial value (Python's "nothing" value)

```python
for path in MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break
```
- **File system check**: `os.path.exists()` checks if file exists
- **Early exit**: `break` stops loop once a model is found
- **Result**: `MODEL_PATH` contains the first existing file, or stays `None`

### MediaPipe Initialization (Lines 113-118)

```python
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
```
- **BaseOptions**: Configuration object for MediaPipe
- `model_asset_path`: Path to the `.task` model file
- This tells MediaPipe where to find the neural network weights

```python
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
```
- **PoseLandmarkerOptions**: Specific settings for pose detection
- `base_options`: Links to the model file
- `output_segmentation_masks=False`: Don't generate body segmentation (saves computation)

```python
detector = vision.PoseLandmarker.create_from_options(options)
```
- **Factory method**: Creates a ready-to-use pose detector object
- This object will process images and return pose landmarks

---

## Part 3: Data Structures (Lines 120-160)

### POSE_CONNECTIONS (Lines 122-143)
```python
POSE_CONNECTIONS = [
    (0, 1), (1, 2), ...  # Face connections
    (11, 13), ...        # Arm connections
]
```
- **List of tuples**: Each `(start_idx, end_idx)` defines a bone/joint connection
- These are hardcoded based on MediaPipe's 33-landmark skeleton structure
- Example: `(11, 13)` means "draw line from landmark 11 to landmark 13"

### LANDMARKS Dictionary (Lines 147-160)
```python
LANDMARKS = {
    'LEFT_SHOULDER': 11,
    'LEFT_ELBOW': 13,
    ...
}
```
- **Dictionary mapping**: Human-readable names → landmark indices
- Makes code readable: `LANDMARKS['LEFT_ELBOW']` instead of `13`
- MediaPipe has 33 landmarks (0-32), but we only track key joints

---

## Part 4: Video Capture Setup (Lines 162-170)

```python
cap = cv2.VideoCapture(0)
```
- **OpenCV video capture**: `0` = default webcam (first camera)
- Returns a `VideoCapture` object that can read frames

```python
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)
```
- **Error handling**: Checks if camera opened successfully
- `exit(1)`: Exits program with error code 1 (failure)

---

## Part 5: Main Processing Loop (Lines 172-280)

### Frame Capture (Lines 173-175)
```python
ret, frame = cap.read()
if not ret:
    break
```
- **`cap.read()`**: Captures one frame from camera
- Returns tuple: `(success_boolean, image_array)`
- `ret`: `True` if frame captured, `False` if error/end of stream
- `frame`: NumPy array of shape `(height, width, 3)` representing the image
- **Early exit**: If capture fails, exit loop

### Image Preprocessing (Lines 177-182)
```python
frame = cv2.flip(frame, 1)
```
- **Horizontal flip**: `1` = flip around vertical axis (mirror effect)
- Makes it feel natural (like a mirror)

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
- **Color space conversion**: OpenCV uses BGR (Blue-Green-Red), MediaPipe needs RGB
- `cvtColor()` converts between color formats

```python
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
```
- **MediaPipe Image object**: Wraps NumPy array in MediaPipe's format
- `SRGB`: Standard RGB color space
- `data`: The actual image array

### Pose Detection (Lines 184-188)
```python
detection_result = detector.detect(mp_image)
```
- **Run neural network**: Processes image through the pose detection model
- Returns a `PoseLandmarkerResult` object containing:
  - `pose_landmarks`: List of detected poses (can detect multiple people)
  - `segmentation_masks`: (if enabled)

```python
if detection_result.pose_landmarks:
    landmarks = detection_result.pose_landmarks[0]
```
- **Check if pose detected**: `pose_landmarks` is `None` if no person found
- `[0]`: Get first person's landmarks (index 0)
- `landmarks`: List of 33 landmark objects, each with `.x`, `.y`, `.z`, `.visibility`

### Drawing & Angle Calculation (Lines 190-274)

**Lines 193-195: Setup**
```python
h, w, _ = frame.shape
y_offset = 30
line_height = 25
```
- Get frame dimensions for coordinate conversion
- `y_offset`: Starting Y position for text (30 pixels from top)
- `line_height`: Spacing between text lines (25 pixels)

**Lines 199-210: Left Elbow Angle Example**
```python
if (LANDMARKS['LEFT_SHOULDER'] < len(landmarks) and 
    LANDMARKS['LEFT_ELBOW'] < len(landmarks) and 
    LANDMARKS['LEFT_WRIST'] < len(landmarks)):
```
- **Safety check**: Ensures all 3 landmarks exist before calculating
- Prevents crashes if body part is occluded (hidden)

```python
left_elbow_angle = calculate_angle(
    landmarks[LANDMARKS['LEFT_SHOULDER']],  # Point A
    landmarks[LANDMARKS['LEFT_ELBOW']],     # Point B (vertex)
    landmarks[LANDMARKS['LEFT_WRIST']]      # Point C
)
```
- **Array indexing**: `landmarks[11]` gets the 11th landmark (left shoulder)
- **Function call**: Calculates angle at elbow between shoulder-elbow-wrist

```python
cv2.putText(frame, f'Left Elbow: {int(left_elbow_angle)}°', 
           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
```
- **Text rendering**: `cv2.putText(image, text, position, font, scale, color, thickness)`
- `f'Left Elbow: {int(left_elbow_angle)}°'`: F-string formatting (Python 3.6+)
- `int()`: Converts float to integer (removes decimals)
- `(10, y_offset)`: Position (x=10, y=30 initially)
- `(255, 255, 255)`: White color in BGR
- `2`: Text thickness

```python
y_offset += line_height
```
- **Increment Y position**: Moves next text down by 25 pixels
- Prevents text from overlapping

### Display & Exit (Lines 276-280)
```python
cv2.imshow('MediaPipe Pose - Joint Angle Measurement', frame)
```
- **Display window**: Shows the processed frame in a window
- Window title: 'MediaPipe Pose - Joint Angle Measurement'
- Updates every frame (creates video effect)

```python
if cv2.waitKey(10) & 0xFF == ord('q'):
    break
```
- **Keyboard input check**: `cv2.waitKey(10)` waits 10ms for keypress
- `& 0xFF`: Bitwise AND to get last 8 bits (handles cross-platform issues)
- `ord('q')`: Converts 'q' character to ASCII code (113)
- **Exit condition**: If 'q' pressed, break out of loop

---

## Key Technical Concepts

### 1. **Coordinate Systems**
- **Normalized (0.0-1.0)**: MediaPipe's output (independent of image size)
- **Pixel coordinates**: Actual screen positions (0 to width/height)
- Conversion: `pixel_x = normalized_x * image_width`

### 2. **Image Representation**
- Images are NumPy arrays: `shape = (height, width, channels)`
- BGR vs RGB: OpenCV uses BGR, most libraries use RGB
- Direct modification: Drawing functions modify the array in-place

### 3. **MediaPipe Tasks API Flow**
1. Create options object
2. Create detector from options
3. Convert image to MediaPipe format
4. Call `detector.detect()`
5. Extract landmarks from result

### 4. **Error Handling Pattern**
- Check if data exists before using it
- Use bounds checking (`< len(landmarks)`)
- Early returns/exits on critical failures

### 5. **Real-time Processing**
- Loop captures → processes → displays frames continuously
- `waitKey(10)` allows UI to update and check for input
- Each iteration processes one frame (~30-60 FPS depending on hardware)
