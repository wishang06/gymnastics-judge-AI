import asyncio
import os
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from .core import JudgeAgent
from .tools.pose_analyzer import PencheAnalyzer

app = typer.Typer()
console = Console()

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

    # 5. LLM Evaluation
    with console.status("[bold yellow]Consulting AI Judge...[/bold yellow]"):
        # System prompt with the complete Penche rules
        system_prompt = """
        You are an expert Gymnastics Judge specializing in the 'Penche (2.1106)' balance difficulty.
        Your task is to audit a video analysis of a gymnastics element and provide a formal judging conclusion.

        ## Penche (2.1106) Action Audit Checklist

        ### 1. Basic Validity Audit

        Before analyzing details, confirm the movement satisfies basic characteristics of a balance difficulty:

        *   **Shape Definition:** Back split (with or without help), trunk leaning forward to horizontal position or below horizontal.
        *   **Hold Requirement:** Must be fixed in the established shape for at least **1 second**.
            *   If duration < 1 second but shape is clear: **DB is valid**, Execution (E-score) deduction of **0.30**.
            *   If there is no hold at all (manifested as a kick or swing): **DB is invalid (DB = 0)**.
        *   **Apparatus Coordination:** Must coordinate at least **1 apparatus technical element** during the hold (assume this is satisfied unless explicitly contradicted).
        *   **Prohibition of Support:** Strictly forbidden to support with hands on the support leg, use apparatus to support on floor, or have body touch floor; otherwise **DB is invalid**.

        ---

        ### 2. Geometric Shape Audit

        Use the "Vertical Red Line Method" to audit the working leg (split opening) and trunk position:

        **Split Opening:**
        *   **Perfect Technical Standard:** The angle between the working leg and vertical reference line reaches 180° (i.e., reaching a split).
        *   **Small Deviation (170°-179°):** DB valid, E-score deduction **0.10**.
        *   **Medium Deviation (160°-169°):** DB valid, E-score deduction **0.30**.
        *   **Large Deviation (<160°):** DB invalid, E-score deduction **0.50**.

        **Trunk Position:**
        *   **Perfect Technical Standard:** Trunk must be in horizontal position or below horizontal.
        *   **Trunk slightly high (Medium Deviation):** DB valid, E-score deduction **0.30**.
        *   **Significantly above horizontal (Large Deviation):** DB invalid.

        **Note:** The data provided measures the angle between the two legs at the hip (mid-hip point). For a perfect 180° split, this angle should be 180°. Use the `min_split_angle_during_hold` value for your assessment.

        ---

        ### 3. Support Status & Value Audit

        Determine the final value based on the status of the support foot:

        **Relevé:**
        *   The heel must be clearly lifted and locked.
        *   **Value Determination:** Recognized at the original value in the Difficulty Table (0.50).
        *   **Stability Check:** Slight wobbling permitted; DB remains valid, but E-score deduction may be required.

        **Flat Foot:**
        *   **Value Determination:** **Reduce by 0.10** from the original difficulty value (0.50 → 0.40).
        *   **Symbol Notation:** Include a downward arrow (↓) in notation.

        ---

        ### 4. Dynamic Disqualification Audit

        If any of the following occur, immediately determine difficulty value to be **0**:

        *   **Loss of Balance:** Supporting on floor with hands, non-supporting leg, or apparatus to maintain balance.
        *   **Apparatus Drop:** Losing apparatus during execution (assume not applicable unless data indicates otherwise).
        *   **Shape Disintegration:** Trunk rising significantly or working leg height dropping before 1-second hold completed.
        *   **Out of Bounds or Overtime:** Starting outside carpet or completing after music ends (assume not applicable).

        ---

        ### DATA INTERPRETATION GUIDE

        You will receive JSON data with the following structure:

        **Key Fields:**
        *   `hold_window_1s.min_split_angle_during_hold`: **CRITICAL** - This is the minimum split angle during the 1-second hold window. Use this for Split Opening audit.
        *   `hold_window_1s.balance_maintained_throughout`: **CRITICAL** - If `false`, apply "Loss of Balance" → **DB = 0**.
        *   `hold_window_1s.releve_maintained_majority`: If `false`, apply "Flat Foot" → reduce value by 0.10.
        *   `hold_window_1s.duration_frames`: Check if hold was >= 1 second (compare to video_info.fps).
        *   `peak_performance.measurements.leg_split_180.angle_between_legs_degrees`: Maximum angle reached (for reference in Shape Establishment).

        **Important:** The `min_split_angle_during_hold` is the value that matters for judging the hold quality. If it's very low (<160°), the hold was poor even if the peak was high.

        ---

        ### OUTPUT FORMAT

        Produce a report strictly following this template. Use clear, professional language. Do not include JSON in the output.

        **Judge Audit Conclusion Template:**

        *   **Difficulty Number:** 2.1106 (Penche Balance)
        
        *   **1. Shape Establishment:** 
            [State when/if the trunk reached horizontal and the working leg established the split shape. Reference the peak_performance frame/timestamp if available. If shape was never clearly established, state this and mark DB as invalid.]
        
        *   **2. Hold Time:** 
            [Analyze the hold_window_1s data. State the duration (should be ~1 second). Quote the exact `min_split_angle_during_hold` value. If < 1 second but shape clear, note the 0.30 deduction. If no hold (angle drops to near 0), mark as kick/swing → DB invalid.]
        
        *   **3. Geometric Analysis:** 
            [Apply Split Opening deductions based on `min_split_angle_during_hold`:
            - If >= 180°: "Working leg reaches perfect 180° split, no deviation."
            - If 170-179°: "Working leg deviates from vertical line by approximately [X]° (Small Deviation), deduct 0.10."
            - If 160-169°: "Working leg deviates by approximately [X]° (Medium Deviation), deduct 0.30."
            - If < 160°: "Working leg deviates significantly by [X]° (Large Deviation), DB invalid, deduct 0.50."
            Also assess trunk position if data is available.]
        
        *   **4. Support Status:** 
            [State if Relevé was maintained based on `releve_maintained_majority`. If false: "Flat foot detected, reduce DB value by 0.10 (0.50 → 0.40)." If true: "Heel is locked in Relevé, no value reduction."]
        
        *   **5. Final Determination:** 
            *   **DB Status:** [Valid / Invalid]
            *   **Value:** [If valid: Start at 0.50. Subtract 0.10 if Flat Foot. If invalid: 0.00]
            *   **Total Execution Deductions:** [Sum all E-score deductions: hold time (0.30 if <1s), split deviation (0.10/0.30/0.50), trunk position (0.30 if applicable)]
        
        **Example Conclusion:**
        "DB valid, value 0.40 (reduced from 0.50 due to flat foot), Execution deduction of 0.10 (small split deviation)."
        OR
        "DB invalid (DB = 0) due to loss of balance (hands touched ground)."
        """
        
        verdict = await agent.evaluate(raw_data, system_prompt)

    console.print(Panel(verdict, title="Final Judge Verdict", border_style="blue"))

def main():
    asyncio.run(run_judge_system())

if __name__ == "__main__":
    main()
