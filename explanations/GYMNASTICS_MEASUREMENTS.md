# Gymnastics Judging Measurements

This document explains the three criteria measurements provided by the video analyzer for gymnastics pose assessment.

## Overview

The tool measures three key criteria:
1. **Leg Split at 180°** - Vertical split measurement
2. **Balance (No Hands on Ground)** - Balance maintenance check
3. **Releve (Tip-toed Feet)** - Foot position verification

## Data Structure

The measurements are returned in a structured format ready for AI API consumption:

```json
{
  "video_info": {
    "total_frames": 900,
    "fps": 30,
    "duration_seconds": 30.0,
    "resolution": "1920x1080"
  },
  "frame_by_frame_data": [
    {
      "frame": 1,
      "timestamp": 0.033,
      "measurements": {
        "leg_split_180": { ... },
        "balance_no_hands_ground": { ... },
        "releve_tip_toed": { ... }
      }
    },
    ...
  ],
  "summary_statistics": { ... }
}
```

---

## 1. Leg Split at 180° (Vertical Split)

### Measurements Provided:

- **`angle_between_legs_degrees`**: The angle formed at the hip between the two legs
  - **Target**: 180° for perfect split
  - **Range**: 0° - 360°
  - **Example**: 175.3° means 4.7° away from perfect

- **`deviation_from_180_degrees`**: How far from perfect 180°
  - **Target**: 0° (perfect)
  - **Lower is better**: 5° or less is considered good
  - **Example**: 4.7° deviation

- **`left_leg_straightness_degrees`**: Knee angle of left leg (hip-knee-ankle)
  - **Target**: 180° (straight leg)
  - **Range**: 0° - 180°
  - **Example**: 178.5° means leg is nearly straight

- **`right_leg_straightness_degrees`**: Knee angle of right leg
  - Same as left leg

- **`lower_foot`**: Which foot is closer to ground
  - Values: `"left"` or `"right"`
  - Used for measuring "from the lower foot" as specified

- **`is_perfect_180`**: Boolean indicating if within 5° tolerance
  - `true` if deviation < 5°
  - `false` otherwise

### How to Interpret:

- **Perfect split**: `angle_between_legs_degrees` ≈ 180°, `deviation_from_180_degrees` < 5°
- **Both legs straight**: Both `left_leg_straightness_degrees` and `right_leg_straightness_degrees` ≈ 180°
- **Good form**: Deviation < 10°, both legs > 170° straightness

---

## 2. Balance (No Hands on Ground)

### Measurements Provided:

- **`left_wrist_y_normalized`**: Vertical position of left wrist (0.0-1.0)
  - **Lower values** = higher on screen (better)
  - **Higher values** = lower on screen (closer to ground)

- **`right_wrist_y_normalized`**: Vertical position of right wrist

- **`ground_reference_y_normalized`**: Reference point for "ground level"
  - Based on the lower ankle position
  - Wrists should be above this value

- **`left_hand_touching`**: Boolean - is left hand touching ground?
  - `true` = hand is at/below ground level
  - `false` = hand is above ground (good)

- **`right_hand_touching`**: Boolean - is right hand touching ground?

- **`any_hand_touching`**: Boolean - are either hands touching?
  - `true` = balance lost (hands down)
  - `false` = balance maintained (good)

- **`balance_maintained`**: Boolean - inverse of `any_hand_touching`
  - `true` = balance maintained
  - `false` = balance lost

### How to Interpret:

- **Perfect balance**: `balance_maintained` = `true` for all frames
- **Balance lost**: `any_hand_touching` = `true` (hands touched ground)
- **Good form**: Wrists stay well above `ground_reference_y_normalized`

---

## 3. Releve (Tip-toed Feet)

### Measurements Provided:

- **`left_leg_vertical_angle_degrees`**: Angle of left leg from vertical
  - **Target**: 0° (perfectly vertical)
  - **Range**: 0° - 90°
  - **Lower is better**: Leg should be straight up

- **`right_leg_vertical_angle_degrees`**: Angle of right leg from vertical

- **`left_knee_angle_degrees`**: Knee joint angle (hip-knee-ankle)
  - **Target**: 180° (straight leg)
  - **Range**: 0° - 180°
  - **Example**: 179.2° means leg is nearly straight

- **`right_knee_angle_degrees`**: Knee joint angle for right leg

- **`left_ankle_raised`**: Boolean - is left ankle raised (heel off ground)?
  - `true` = ankle is raised (releve position)
  - `false` = flat foot

- **`right_ankle_raised`**: Boolean - is right ankle raised?

- **`both_feet_releve`**: Boolean - are both feet in releve?
  - `true` = both feet tip-toed (good)
  - `false` = at least one foot flat

### How to Interpret:

- **Perfect releve**: 
  - `both_feet_releve` = `true`
  - Both `knee_angle_degrees` ≈ 180° (straight legs)
  - Both `leg_vertical_angle_degrees` < 10° (nearly vertical)

- **Good form**: 
  - Both feet raised
  - Legs straight (knee angles > 170°)
  - Legs vertical (vertical angles < 15°)

---

## Summary Statistics

The tool also provides aggregated statistics across all frames:

```json
{
  "summary_statistics": {
    "leg_split": {
      "average_angle": 175.3,
      "average_deviation_from_180": 4.7,
      "min_angle": 165.2,
      "max_angle": 179.8
    },
    "balance": {
      "frames_with_hands_touching": 5,
      "total_frames_analyzed": 900,
      "balance_percentage": 99.4
    },
    "releve": {
      "frames_in_releve": 850,
      "total_frames_analyzed": 900,
      "releve_percentage": 94.4
    }
  }
}
```

---

## Usage Example

### Command Line:
```bash
# Analyze video and export to JSON
python video_analyzer.py video.mp4 -j measurements.json

# Or with video output
python video_analyzer.py video.mp4 -s -j measurements.json
```

### Python Code:
```python
from video_analyzer import analyze_video, export_measurements_to_json

# Analyze video
result = analyze_video("gymnastics_pose.mp4", save_output=True)

# Export to JSON for AI API
export_measurements_to_json(result, "measurements.json")

# Access specific measurements
for frame_data in result['frame_by_frame_data']:
    frame_num = frame_data['frame']
    measurements = frame_data['measurements']
    
    # Check leg split
    split_angle = measurements['leg_split_180']['angle_between_legs_degrees']
    deviation = measurements['leg_split_180']['deviation_from_180_degrees']
    
    # Check balance
    balance_ok = measurements['balance_no_hands_ground']['balance_maintained']
    
    # Check releve
    releve_ok = measurements['releve_tip_toed']['both_feet_releve']
    
    print(f"Frame {frame_num}: Split={split_angle:.1f}°, Balance={balance_ok}, Releve={releve_ok}")
```

---

## Notes for AI API Integration

1. **Normalized Coordinates**: Y-coordinates are normalized (0.0-1.0), where:
   - `0.0` = top of frame
   - `1.0` = bottom of frame
   - Higher values = lower on screen (closer to ground)

2. **Angle Measurements**: All angles are in degrees (0-360° range)

3. **Boolean Values**: Use `true`/`false` for pass/fail criteria

4. **Frame-by-Frame Data**: Each frame has a `timestamp` field for temporal analysis

5. **Missing Data**: Some measurements may be `null` if body parts are not detected (occluded)

6. **Tolerance Values**: 
   - Leg split: 5° tolerance for "perfect"
   - Balance: 2% frame height tolerance for ground detection
   - Releve: 1% tolerance for ankle position

---

## Scoring Suggestions for AI API

When sending this data to an AI API, you might want to calculate scores like:

- **Leg Split Score**: `100 - (deviation_from_180 * 2)` (max 100, min 0)
- **Balance Score**: `balance_percentage` (0-100)
- **Releve Score**: `releve_percentage` (0-100)
- **Overall Score**: Weighted average of the three criteria

Example prompt for AI API:
```
Analyze this gymnastics pose data:
- Leg split: {angle_between_legs}° (deviation: {deviation}°)
- Balance: {balance_percentage}% maintained
- Releve: {releve_percentage}% of frames

Provide detailed feedback on form, technique, and areas for improvement.
```
