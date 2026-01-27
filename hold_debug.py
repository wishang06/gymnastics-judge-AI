import asyncio
import json
import os

from src.gymnastics_judge.tools.pose_analyzer import PencheAnalyzer


async def main():
    # Map menu IDs used by the CLI to filenames in /videos
    # This is just a helper for debugging hold time quickly.
    videos = {
        "9": os.path.join("videos", "Perfect-1.mp4"),
        "10": os.path.join("videos", "Perfect-2.mp4"),
    }

    analyzer = PencheAnalyzer(show_video=False)

    for key, path in videos.items():
        if not os.path.exists(path):
            print(f"[{key}] Missing: {path}")
            continue
        print(f"\n=== Testing video {key}: {path} ===")
        result = await analyzer.analyze(path)
        hold = result.get("hold_window_1s")
        peak = result.get("peak_performance")
        if peak:
            leg = peak["measurements"]["leg_split_180"]
            print(
                "peak_frame:",
                peak.get("frame"),
                "peak_angle:",
                leg.get("angle_between_legs_degrees"),
                "knee:",
                leg.get("angle_knee_degrees"),
                "ankle:",
                leg.get("angle_ankle_degrees"),
                "raw360:",
                leg.get("angle_between_legs_degrees_raw_360"),
            )
        else:
            print("peak_frame: None")
        print("hold_window_1s:", json.dumps(hold, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

