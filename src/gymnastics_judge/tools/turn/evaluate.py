"""
FIG D/E scoring for Back Attitude Pivot (3.1203).
Wide gate (leg > 45°) for counting; buffer -150°; angle +7.5° compensation.
D valid if angle >= 70° and turns >= 1; E deductions by angle band.
"""
import os
from . import config


def evaluate_turn(video_path: str, analysis_data: list) -> dict:
    """
    Returns dict with: report (markdown text), d_score, e_deduction, final_score,
    is_d_valid, valid_turns, final_official_angle, status_msg, d_reason, e_reason.
    """
    rules = config.rules_config
    counting_threshold = config.THRESHOLDS["MOVEMENT_START_ANGLE"]
    passing_grade = rules["shape_requirements"]["valid_min_angle"]
    buffer_deg = config.ROTATION_BUFFER_DEG
    comp_deg = config.ANGLE_COMPENSATION_DEG

    raw_valid_degrees = 0.0
    valid_angles = []
    prev_total_rotation = 0

    for frame in analysis_data:
        current_total = frame["total_rotation"]
        current_leg_angle = frame["thigh_lift_angle"]
        delta = abs(current_total - prev_total_rotation)
        if current_leg_angle > counting_threshold:
            raw_valid_degrees += delta
            valid_angles.append(current_leg_angle)
        prev_total_rotation = current_total

    final_valid_degrees = max(0, raw_valid_degrees - buffer_deg)
    valid_turns_float = final_valid_degrees / 360.0
    valid_turns = int(valid_turns_float)

    avg_angle_raw = sum(valid_angles) / len(valid_angles) if valid_angles else 0
    final_official_angle = avg_angle_raw + comp_deg if avg_angle_raw > 0 else 0

    d_score = 0.0
    status_msg = ""
    d_reason = ""
    is_d_valid = False
    angle_pass = final_official_angle >= passing_grade
    turns_pass = valid_turns >= 1

    if angle_pass and turns_pass:
        is_d_valid = True
        status_msg = "VALID"
        base = rules["difficulty_value"]
        bonus = (valid_turns - 1) * rules["bonus_per_rotation"] if valid_turns > 1 else 0.0
        d_score = base + bonus
        d_reason = f"Base {base:.2f}" + (f" + Bonus {bonus:.2f} ({valid_turns} turns)" if bonus > 0 else f" ({valid_turns} turn)")
    else:
        status_msg = "INVALID / NO VALUE"
        reasons = []
        if not angle_pass:
            reasons.append(f"Angle {final_official_angle:.1f}° < 70°")
        if not turns_pass:
            reasons.append(f"Turns {valid_turns} < 1")
        d_reason = " & ".join(reasons)

    if final_official_angle >= 90:
        e_deduction = 0.00
        e_reason = "Perfect shape (>90°), no deduction."
    elif final_official_angle >= 80:
        e_deduction = 0.10
        e_reason = "Small deviation (80°-89°)."
    elif final_official_angle >= 70:
        e_deduction = 0.30
        e_reason = "Medium deviation (70°-79°)."
    else:
        e_deduction = 0.50
        e_reason = "Large deviation (<70°)."

    final_score = max(0, d_score - e_deduction)

    report = f"""
# Analysis Report: {os.path.basename(video_path)}

## 1. Performance Data
* **Raw Rotation**: {raw_valid_degrees:.2f}°
* **Adjusted Rotation**: {final_valid_degrees:.2f}° (Buffer -{buffer_deg}°)
* **Calculated Turns**: {valid_turns}
* **Average Thigh Angle**: {final_official_angle:.2f}° (includes +{comp_deg}° compensation)

## 2. D-Score (Difficulty)
* **Status**: {status_msg}
* **Score**: {d_score:.2f}
* **Reason**: {d_reason}

## 3. E-Score (Execution)
* **Deduction**: -{e_deduction:.2f}
* **Reason**: {e_reason}

## 4. Final Verdict
* **Final Score**: {final_score:.2f}
"""
    return {
        "report": report.strip(),
        "d_score": d_score,
        "e_deduction": e_deduction,
        "e_reason": e_reason,
        "final_score": final_score,
        "is_d_valid": is_d_valid,
        "valid_turns": valid_turns,
        "final_official_angle": final_official_angle,
        "status_msg": status_msg,
        "d_reason": d_reason,
        "raw_valid_degrees": raw_valid_degrees,
        "final_valid_degrees": final_valid_degrees,
    }
