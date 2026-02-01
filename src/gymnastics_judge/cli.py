import asyncio
import json
import os
import re
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.syntax import Syntax
from rich.table import Table
from .core import JudgeAgent
from .tools.pose_analyzer import PencheAnalyzer

app = typer.Typer()
console = Console()


def _format_verdict(verdict: str):
    """Extract JSON from markdown code fence, round numbers, return Rich Syntax or None."""
    text = verdict.strip()
    # Strip ```json ... ``` or ``` ... ```
    match = re.search(r"^```(?:json)?\s*\n(.*?)```\s*$", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    def round_floats(obj, ndigits=1):
        if isinstance(obj, float):
            return round(obj, ndigits)
        if isinstance(obj, dict):
            return {k: round_floats(v, ndigits) for k, v in obj.items()}
        if isinstance(obj, list):
            return [round_floats(v, ndigits) for v in obj]
        return obj

    rounded = round_floats(data)
    pretty = json.dumps(rounded, indent=2)
    return Syntax(pretty, "json", theme="monokai", line_numbers=False)

async def run_judge_system():
    console.print(Panel.fit("[bold magenta]AI Gymnastics Judge System[/bold magenta]", subtitle="Powered by Gemini 3 Flash"))

    # 1. Initialize Agent & Tools
    try:
        agent = JudgeAgent()
        analyzer = PencheAnalyzer()
        tools = {
            "1": analyzer
        }
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        return

    # 2. Select Video
    video_dir = "videos"
    if not os.path.exists(video_dir):
        video_dir = "."
    
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    
    if not mp4_files:
        console.print("[red]No .mp4 files found in 'videos' folder![/red]")
        return

    table = Table(title="Available Videos")
    table.add_column("ID", style="cyan")
    table.add_column("Filename", style="green")
    
    for i, f in enumerate(mp4_files):
        table.add_row(str(i+1), f)
    
    console.print(table)
    
    vid_idx = IntPrompt.ask("Select video ID", choices=[str(i+1) for i in range(len(mp4_files))])
    selected_video = os.path.join(video_dir, mp4_files[vid_idx-1])

    # 3. Select Tool (Only 1 for now, but expandable)
    console.print("\n[bold]Select Analysis Tool:[/bold]")
    console.print("1. Penche Analyzer (2.1106)")
    
    tool_choice = Prompt.ask("Select tool ID", choices=["1"], default="1")
    selected_tool = tools[tool_choice]

    # 4. Run Analysis
    with console.status(f"[bold green]Running {selected_tool.name}...[/bold green]"):
        try:
            raw_data = await selected_tool.analyze(selected_video)
        except Exception as e:
            console.print(f"[red]Analysis failed:[/red] {e}")
            return

    console.print("[green]✓ Analysis Complete[/green]")
    
    if raw_data.get('peak_performance'):
        angle = raw_data['peak_performance']['measurements']['leg_split_180']['angle_between_legs_degrees']
        console.print(f"Peak Split Angle: {angle:.1f}°")
    else:
        console.print("[yellow]! No peak performance detected (no pose found?)[/yellow]")
        return

    # 5. LLM Vision Review (Hand Support) — bypass MediaPipe hand detection
    hold_window = raw_data.get("hold_window_1s") or {}
    with console.status("[bold yellow]Reviewing hand support (Gemini vision)...[/bold yellow]"):
        try:
            hand_review = await agent.review_hand_support(selected_video, hold_window)
        except Exception as e:
            console.print(f"[red]Hand-support review failed:[/red] {e}")
            hand_review = {
                "hand_support_detected": None,
                "confidence": 0.0,
                "notes": f"Hand-support review failed: {e}",
                "evidence_frame_numbers": [],
                "used_model": None,
                "used_frames": {},
            }

    raw_data["llm_hand_support"] = hand_review
    # Incorporate the new workflow:
    # - We consider the "hold" to begin when hands are OFF during the CV-defined hold segment.
    # - The LLM provides an estimate of the longest hands-off run inside the review window.
    if isinstance(hand_review.get("balance_maintained_during_hold_estimate"), bool):
        hold_window["balance_maintained_throughout"] = bool(hand_review["balance_maintained_during_hold_estimate"])
        hold_window["balance_source"] = "llm_video_review"
        if isinstance(hand_review.get("hands_off_longest_run"), dict):
            hold_window["hands_off_longest_run"] = hand_review["hands_off_longest_run"]
            # Prefer this duration for time judging, because rule timing starts when hands are off.
            hold_window["duration_seconds_hands_off_estimate"] = float(
                hand_review["hands_off_longest_run"].get("duration_seconds_estimate") or 0.0
            )
    raw_data["hold_window_1s"] = hold_window

    hs_hold = hand_review.get("hand_support_detected_during_hold_estimate")
    if hs_hold is True:
        console.print("[red]Hand support detected during hold (LLM review).[/red]")
    elif hs_hold is False:
        console.print("[green]No hand support detected during hold (LLM review).[/green]")
    else:
        console.print("[yellow]Hand support during hold could not be determined (LLM review).[/yellow]")

    # 6. LLM Vision Review (Relevé) — bypass MediaPipe foot detection
    with console.status("[bold yellow]Reviewing relevé (Gemini vision)...[/bold yellow]"):
        try:
            releve_review = await agent.review_releve(selected_video, hold_window)
        except Exception as e:
            console.print(f"[red]Relevé review failed:[/red] {e}")
            releve_review = {
                "releve_maintained_majority_estimate": None,
                "confidence": 0.0,
                "notes": f"Relevé review failed: {e}",
                "flat_evidence_frame_numbers": [],
                "frame_statuses": [],
                "used_model": None,
                "used_frames": {},
            }

    raw_data["llm_releve"] = releve_review
    if isinstance(releve_review.get("releve_maintained_majority_estimate"), bool):
        hold_window["releve_maintained_majority"] = bool(releve_review["releve_maintained_majority_estimate"])
        hold_window["releve_source"] = "llm_video_review"
    raw_data["hold_window_1s"] = hold_window

    rel = releve_review.get("releve_maintained_majority_estimate")
    if rel is True:
        console.print("[green]Relevé maintained during hold (LLM review).[/green]")
    elif rel is False:
        console.print("[red]Flat foot / no consistent relevé during hold (LLM review).[/red]")
    else:
        console.print("[yellow]Relevé could not be determined (LLM review).[/yellow]")

    # 7. LLM Evaluation (Rules + CV stats + LLM hand-support + LLM relevé)
    with console.status("[bold yellow]Consulting AI Judge...[/bold yellow]"):
        # System prompt for Penche judging (D/E scoring)
        system_prompt = """
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
          (This is NOT computed by MediaPipe in this project.)
        - If `llm_hand_support.hand_support_detected_during_hold_estimate` is null/unknown: mark hand-support as "Not measured" and apply no
          disqualification (DB unchanged, E=0.0), but note the uncertainty.

        Other disqualifiers mentioned in the rulebook (knee support, falling) are NOT measured by the CV tool.
        Do NOT invent them. Mark as \"Not measured\" and apply 0 deduction for those items.

        Stability deductions (minor/s significant movement) are NOT measured by the CV tool. Do NOT invent them.
        Mark as \"Not measured\" and apply 0 deduction.

        ### D) Support Foot / Relevé
        Relevé is judged by the Gemini vision reviewer, not MediaPipe:
        - `llm_releve.releve_maintained_majority_estimate` (boolean or null)

        This is NOT part of the D/E breakdown above, but we still report it:
        - If false, note: \"Flat foot / no consistent relevé detected\".
        (Do not apply extra deductions unless explicitly requested by the D/E breakdown rules.)

        ---

        ## Required Output (STRICT FORMAT)
        Output exactly these sections and keys every time:

        ### Judge Summary
        - **Element**: Penche (2.1106)
        - **DB (D-score contribution)**: 0.5 or 0
        - **Total E-score deduction**: a cumulative negative number, calculated by adding up the E deductions from the rules above.

        ### Computed Metrics (from CV JSON)
        - **hold_seconds**: <number> (use `hold_window_1s.duration_seconds_hands_off_estimate` if present, else `hold_window_1s.duration_seconds`)
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
        - Do not mention \"training data\".
        """
        
        verdict = await agent.evaluate(raw_data, system_prompt)

    renderable = _format_verdict(verdict)
    if renderable is not None:
        console.print(Panel(renderable, title="Final Judge Verdict", border_style="blue"))
    else:
        # Fallback: strip code fence for plain text
        fallback = re.sub(r"^```(?:json)?\s*\n?", "", verdict.strip())
        fallback = re.sub(r"\n?```\s*$", "", fallback)
        console.print(Panel(fallback or verdict, title="Final Judge Verdict", border_style="blue"))

def main():
    asyncio.run(run_judge_system())

if __name__ == "__main__":
    main()
