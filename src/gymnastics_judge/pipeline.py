"""
Shared pipeline: run single analysis (Penche or rhythmic) and return verdict + reports.
Used by both CLI and web app.
"""
import asyncio
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _extract_d_e_scores(simple_report: Optional[str], verdict: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Extract D and E score strings from simple_report and verdict for display. Returns (d_score, e_score)."""
    text = " ".join(filter(None, [simple_report or "", verdict or ""]))
    d_score: Optional[str] = None
    e_score: Optional[str] = None
    d_m = re.search(r"(?:D分|难度分|DB|D-score)[：:\s]*(有效|无效|[0-5](?:\.\d+)?)", text, re.I)
    if d_m:
        d_score = d_m.group(1)
    if not d_score:
        d_m = re.search(r"难度分为?\s*([0-5](?:\.\d+)?)", text)
        if d_m:
            d_score = d_m.group(1)
    if not d_score:
        d_m = re.search(r"难度分[^0-9]*([0-5](?:\.\d+)?)\s*分", text)
        if d_m:
            d_score = d_m.group(1)
    if not d_score:
        d_m = re.search(r"(?:D分|难度分)[^\n]*?([0-5](?:\.\d+)?)\s*分", text)
        if d_m:
            d_score = d_m.group(1)
    e_m = re.search(r"(?:E分|扣分|total e-score|e-score deduction|总计扣分)[：:\s]*\s*([－\-]?\d*\.?\d+)", text, re.I)
    if e_m:
        e_score = e_m.group(1).replace("－", "-")
    if not e_score:
        e_m = re.search(r"(?:E分|E-score)[^0-9－\-]*([－\-]?\d*\.?\d+)", text, re.I)
        if e_m:
            e_score = e_m.group(1).replace("－", "-")
    if not e_score:
        e_m = re.search(r"扣分[^0-9－\-]*([－\-]?\d*\.?\d+)", text)
        if e_m:
            e_score = e_m.group(1).replace("－", "-")
    if not e_score:
        e_m = re.search(r"(?:E|扣)[^\n]*?([－\-]0\.\d+)", text)
        if e_m:
            e_score = e_m.group(1).replace("－", "-")
    if not e_score:
        e_m = re.search(r"([－\-]0\.\d+)", text)
        if e_m:
            e_score = e_m.group(1).replace("－", "-")
    return (d_score, e_score)

from .tools.pose_analyzer import PencheAnalyzer
from .tools.rhythmic_element_analyzer import RhythmicElementAnalyzer

VIDEO_CATEGORY_DIRS = ("videos/penche", "videos/1_2096", "videos/1_2105")

PENCHE_SYSTEM_PROMPT = """
You are an expert gymnastics judge. You MUST compute the D-score contribution (DB) and total E-score deductions
for the Penche element (2.1106) using ONLY the provided JSON:
- MediaPipe-derived joint/angle/timing measurements (NO hand detection, NO relevé detection)
- A Gemini vision review result for hand-support during the hold window
- A Gemini vision review result for relevé during the hold window

## Penche: D-Score & E-Score Breakdown (Authoritative Rules)

### A) Angular Deviation (Split Opening)
Use the split opening during the hold window.

Let:
- split_angle_deg = `hold_window_1s.min_split_angle_during_hold`
- effective_angle = clamp the split angle into [0, 180] using:
  effective_angle = min(split_angle_deg, 360 - split_angle_deg) if split_angle_deg > 180 else split_angle_deg
  then effective_angle = min(max(effective_angle, 0), 180)
- deviation_deg = max(0, 180 - effective_angle)

Rules:
- deviation_deg == 0°: D valid (DB = 0.5); (E deduction 0.0)
- 1° to 10° deviation: D valid (DB = 0.5); (E deduction -0.1)
- 11° to 19° deviation: D valid (DB = 0.5); (E deduction -0.3)
- deviation_deg >= 20°: D invalid (DB = 0.0); (E deduction -0.5)

### B) Time (Duration of Hold)
The CV tool defines the hold segment start/end as the best contiguous region near the peak split where the
split angle remains stable (within `hold_window_1s.angle_stability_threshold_deg`) and stays near peak
(within `hold_window_1s.near_peak_tolerance_deg` of `hold_window_1s.peak_effective_split_angle_deg`).

Use the best available hold duration estimate (because timing starts when hands are off):
- If present: hold_seconds = `hold_window_1s.duration_seconds_hands_off_estimate`
- Else: hold_seconds = `hold_window_1s.duration_seconds`

Rules:
- hold_seconds < 1.0s: D invalid (DB = 0.0)
- 0.5s to 1.0s: D valid (DB = 0.5); (E deduction -0.3)
- > 1.0s: D valid (DB = 0.5); (E deduction 0.0)

### C) Disqualification & Stability
D invalid (DB = 0.0) if:
- Hand Support DURING HOLD: if `llm_hand_support.hand_support_detected_during_hold_estimate` is true,
  then (E deduction -0.5) and (DB = 0.0).
- If `llm_hand_support.hand_support_detected_during_hold_estimate` is null/unknown: mark hand-support as "Not measured" and apply no
  disqualification (DB unchanged, E=0.0), but note the uncertainty.

Other disqualifiers mentioned in the rulebook (knee support, falling) are NOT measured by the CV tool.
Do NOT invent them. Mark as "Not measured" and apply 0 deduction for those items.

Stability deductions (minor/s significant movement) are NOT measured by the CV tool. Do NOT invent them.
Mark as "Not measured" and apply 0 deduction.

### D) Support Foot / Relevé
Relevé is judged by the Gemini vision reviewer, not MediaPipe:
- `llm_releve.releve_maintained_majority_estimate` (boolean or null)

This is NOT part of the D/E breakdown above, but we still report it:
- If false, note: "Flat foot / no consistent relevé detected".
(Do not apply extra deductions unless explicitly requested by the D/E breakdown rules.)

---

## Required Output (STRICT FORMAT)
Output exactly these sections and keys every time:

### Judge Summary
- **Element**: Penche (2.1106)
- **DB (D-score contribution)**: 0.5 or 0
- **Total E-score deduction**: a cumulative negative number, calculated by adding up the E deductions from the rules above.

### Computed Metrics (from CV JSON)
- **hold_seconds**: <number>
- **split_angle_deg (raw)**: <number>
- **effective_angle_deg (clamped)**: <number>
- **deviation_deg**: <number>
- **hand_support_detected_during_hold (LLM)**: true/false/null
- **hand_support_confidence (LLM)**: <number>
- **releve_maintained_majority (LLM)**: true/false/null
- **releve_confidence (LLM)**: <number>

### D/E Rule Application
- **Angle rule**: (state which bucket triggered, and the E deduction)
- **Time rule**: (state which bucket triggered, and the E deduction)
- **Disqualification rule (hand support)**: (state if triggered; if triggered DB=0.0 and E=-0.5)
- **Stability/Knee/Fall**: Not measured (E=0.0)

### Final Notes
- 1–3 short bullet comments explaining WHY the deductions happened using the computed metrics.
- Do not mention "training data".
"""


def get_tools() -> Dict[str, Any]:
    """Return dict of tool_id -> analyzer instance."""
    penche = PencheAnalyzer(show_video=False)
    return {
        "1": penche,
        "2": RhythmicElementAnalyzer("1.2096"),
        "3": RhythmicElementAnalyzer("1.2105"),
    }


def list_videos_for_tool(video_dir: str) -> List[str]:
    """List video paths (mp4, mov) in video_dir, sorted. Deduplicated."""
    path = Path(video_dir)
    if not path.exists():
        return []
    seen = set()
    files = []
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        for f in path.glob(ext):
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                files.append(key)
    return sorted([str(f) for f in files], key=lambda x: os.path.basename(x).lower())


def ensure_video_categories() -> None:
    for d in VIDEO_CATEGORY_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)


async def run_single_analysis(
    tool_id: str,
    video_path: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    show_mediapipe_window: bool = False,
) -> Dict[str, Any]:
    """
    Run full analysis for the given tool and video. Returns dict with:
    tool_name, is_penche, video_path, peak_image_path, verdict, simple_report, comprehensive_report, error.
    When show_mediapipe_window is True and tool is Penche, an OpenCV window with pose overlay is shown on the server.
    """
    def progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    tools = get_tools()
    if tool_id not in tools:
        return {"error": f"Unknown tool_id: {tool_id}"}
    selected_tool = tools[tool_id]
    if tool_id == "1" and show_mediapipe_window:
        selected_tool = PencheAnalyzer(show_video=True)
    result = {
        "tool_name": selected_tool.name,
        "is_penche": type(selected_tool).__name__ == "PencheAnalyzer",
        "video_path": video_path,
        "peak_image_path": None,
        "verdict": None,
        "simple_report": None,
        "comprehensive_report": None,
        "error": None,
    }

    progress("分析中…")
    try:
        raw_data = await selected_tool.analyze(video_path)
    except Exception as e:
        result["error"] = str(e)
        return result

    if not result["is_penche"]:
        result["peak_image_path"] = raw_data.get("peak_image")
        from .core import JudgeAgent
        agent = JudgeAgent()
        progress("生成简易动作报告…")
        try:
            result["simple_report"] = (await agent.simple_move_report(selected_tool.name, raw_data)).strip()
        except Exception as e:
            result["simple_report"] = f"生成失败: {e}"
        progress("生成综合报告…")
        try:
            result["comprehensive_report"] = (await agent.comprehensive_report(selected_tool.name, raw_data)).strip()
        except Exception as e:
            result["comprehensive_report"] = f"生成失败: {e}"
        return result

    # Penche: need peak for continuation
    if not raw_data.get("peak_performance"):
        result["error"] = "No peak performance detected (no pose found?)."
        return result

    from .core import JudgeAgent
    agent = JudgeAgent()
    hold_window = raw_data.get("hold_window_1s") or {}

    progress("审查手支撑…")
    try:
        hand_review = await agent.review_hand_support(video_path, hold_window)
    except Exception as e:
        hand_review = {
            "hand_support_detected_during_hold_estimate": None,
            "confidence": 0.0,
            "notes": str(e),
        }
    raw_data["llm_hand_support"] = hand_review
    if isinstance(hand_review.get("balance_maintained_during_hold_estimate"), bool):
        hold_window["balance_maintained_throughout"] = bool(hand_review["balance_maintained_during_hold_estimate"])
        if isinstance(hand_review.get("hands_off_longest_run"), dict):
            hold_window["duration_seconds_hands_off_estimate"] = float(
                hand_review["hands_off_longest_run"].get("duration_seconds_estimate") or 0.0
            )
    raw_data["hold_window_1s"] = hold_window

    progress("审查 relevé…")
    try:
        releve_review = await agent.review_releve(video_path, hold_window)
    except Exception as e:
        releve_review = {
            "releve_maintained_majority_estimate": None,
            "confidence": 0.0,
            "notes": str(e),
        }
    raw_data["llm_releve"] = releve_review
    if isinstance(releve_review.get("releve_maintained_majority_estimate"), bool):
        hold_window["releve_maintained_majority"] = bool(releve_review["releve_maintained_majority_estimate"])
    raw_data["hold_window_1s"] = hold_window

    progress("AI 裁判评分…")
    try:
        verdict = await agent.evaluate(raw_data, PENCHE_SYSTEM_PROMPT)
    except Exception as e:
        verdict = f"Judge evaluation failed: {e}"
    result["verdict"] = verdict

    progress("生成简易动作报告…")
    try:
        result["simple_report"] = (await agent.simple_move_report(selected_tool.name, raw_data, judge_verdict=verdict)).strip()
    except Exception as e:
        result["simple_report"] = f"生成失败: {e}"

    progress("生成综合报告…")
    try:
        result["comprehensive_report"] = (await agent.comprehensive_report(selected_tool.name, raw_data, judge_verdict=verdict)).strip()
    except Exception as e:
        result["comprehensive_report"] = f"生成失败: {e}"

    return result


async def run_overall_analysis(
    items: List[tuple],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    items: list of (tool_id, video_path) for each selected analysis.
    Runs full analysis for each, then generates Overall Person Report (四维度画像 + 运动员深度分析).
    Returns dict with per_element_results (list of single-run results) and overall_report (radar, athlete_profile, practice_guide, full_report, per_element).
    """
    def progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    per_element_results: List[Dict[str, Any]] = []
    for i, (tool_id, video_path) in enumerate(items):
        progress(f"正在分析 {i + 1}/{len(items)}: {video_path}")
        try:
            single = await run_single_analysis(
                tool_id,
                video_path,
                progress_callback=None,
                show_mediapipe_window=False,
            )
        except Exception as e:
            single = {"error": str(e), "tool_name": f"工具 {tool_id}"}
        d_score, e_score = _extract_d_e_scores(
            single.get("simple_report"),
            single.get("verdict"),
        )
        per_element_results.append({
            "tool_name": single.get("tool_name", ""),
            "verdict": single.get("verdict"),
            "simple_report": single.get("simple_report"),
            "comprehensive_report": single.get("comprehensive_report"),
            "d_score": d_score,
            "e_score": e_score,
        })

    progress("正在生成运动员整体报告…")
    from .core import JudgeAgent
    agent = JudgeAgent()
    try:
        overall_report = await agent.overall_person_report(per_element_results)
    except Exception as e:
        overall_report = {
            "per_element": [],
            "radar": {"A": 5, "B": 5, "C": 5, "D": 5},
            "athlete_profile": f"报告生成失败: {e}",
            "practice_guide": "",
            "full_report": "",
        }

    return {
        "per_element_results": per_element_results,
        "overall_report": overall_report,
    }
