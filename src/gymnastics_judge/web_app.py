"""
Flask web app for AI Gymnastics Judge: multi-page flow (title -> select tool -> select file -> run -> reports).
All UI in Chinese. Serves static pages and API for tools, videos, and run.
"""
import asyncio
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Lazy import pipeline so tools are only created when needed
WEB_ROOT = Path(__file__).resolve().parent.parent.parent

app = Flask(
    __name__,
    static_folder=str(WEB_ROOT / "static"),
    static_url_path="",
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "gymnastics-judge-dev-secret")
CORS(app)

TOOLS_CONFIG = [
    {"id": "1", "name": "平衡腿结环 (Penche 2.1106)", "name_en": "Penche (2.1106)"},
    {"id": "2", "name": "交换腿鹿跳结环 (1.2096)"},
    {"id": "3", "name": "跨跳结环 (1.2105)"},
    {"id": "4", "name": "后屈腿转体 (Back Attitude Pivot 3.1203)"},
]

VIDEO_DIRS = {"1": "videos/penche", "2": "videos/1_2096", "3": "videos/1_2105", "4": "videos/3_1203"}


def _list_videos(video_dir: str) -> list:
    path = WEB_ROOT / video_dir
    if not path.exists():
        return []
    seen = set()
    files = []
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        for f in path.glob(ext):
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                files.append(f)
    return sorted(files, key=lambda x: x.name.lower())


@app.route("/styles/<path:filename>")
def serve_styles(filename):
    return send_from_directory(Path(app.static_folder) / "styles", filename)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/select-tool")
def select_tool():
    return send_from_directory(app.static_folder, "select-tool.html")


@app.route("/select-file")
def select_file():
    return send_from_directory(app.static_folder, "select-file.html")


@app.route("/run")
def run_page():
    return send_from_directory(app.static_folder, "run.html")


@app.route("/simple-report")
def simple_report_page():
    return send_from_directory(app.static_folder, "simple-report.html")


@app.route("/comprehensive-report")
def comprehensive_report_page():
    return send_from_directory(app.static_folder, "comprehensive-report.html")


@app.route("/select-overall")
def select_overall():
    return send_from_directory(app.static_folder, "select-overall.html")


@app.route("/run-overall")
def run_overall_page():
    return send_from_directory(app.static_folder, "run-overall.html")


@app.route("/overall-report")
def overall_report_page():
    return send_from_directory(app.static_folder, "overall-report.html")


@app.route("/api/tools", methods=["GET"])
def api_tools():
    return jsonify({"tools": TOOLS_CONFIG})


@app.route("/api/tools/<tool_id>/videos", methods=["GET"])
def api_tool_videos(tool_id):
    if tool_id not in VIDEO_DIRS:
        return jsonify({"error": "Unknown tool"}), 400
    video_dir = VIDEO_DIRS[tool_id]
    base = WEB_ROOT / video_dir
    # Fallback for Penche: list flat videos/ if penche folder empty
    if tool_id == "1":
        files = _list_videos(video_dir)
        if not files:
            files = _list_videos("videos")
    else:
        files = _list_videos(video_dir)
    items = [{"id": i, "path": str(f), "name": f.name} for i, f in enumerate(files)]
    return jsonify({"videos": items})


def _run_pipeline_sync(
    tool_id: str,
    video_path: str,
    show_mediapipe_window: bool = False,
    show_yolo_window: bool = False,
) -> dict:
    from .pipeline import run_single_analysis
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            run_single_analysis(
                tool_id,
                video_path,
                show_mediapipe_window=show_mediapipe_window,
                show_yolo_window=show_yolo_window,
            )
        )
    finally:
        loop.close()


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json() or {}
    tool_id = data.get("tool_id")
    video_id = data.get("video_id")
    if tool_id is None or video_id is None:
        return jsonify({"error": "缺少 tool_id 或 video_id"}), 400
    video_dir = VIDEO_DIRS.get(tool_id)
    if not video_dir:
        return jsonify({"error": "未知工具"}), 400
    files = _list_videos(video_dir)
    if tool_id == "1" and not files:
        files = _list_videos("videos")
    if video_id < 0 or video_id >= len(files):
        return jsonify({"error": "无效的视频序号"}), 400
    video_path = str(files[video_id])
    show_mediapipe_window = data.get("show_mediapipe_window", False) and tool_id == "1"
    show_yolo_window = data.get("show_yolo_window", False) and tool_id == "4"
    try:
        result = _run_pipeline_sync(
            tool_id,
            video_path,
            show_mediapipe_window=show_mediapipe_window,
            show_yolo_window=show_yolo_window,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if result.get("error"):
        return jsonify(result), 422
    # Normalize paths for frontend: relative URLs to serve video and audit image
    def _url_path(local_path: str) -> str:
        return local_path.replace("\\", "/") if local_path else ""

    v_path = result.get("video_path") or ""
    if v_path:
        try:
            v_rel = os.path.relpath(v_path, str(WEB_ROOT))
            v_rel = _url_path(v_rel)
            if ".." not in v_rel and v_rel.startswith("videos/"):
                result["video_url"] = f"/api/files/video?p={v_rel}"
            elif ".." not in v_rel:
                result["video_url"] = f"/api/files/video?p={v_rel}"
        except ValueError:
            pass
    if not result.get("video_url") and v_path:
        result["video_url"] = f"/api/files/video?p=videos/penche/{os.path.basename(v_path)}"

    peak = result.get("peak_image_path")
    if peak:
        try:
            p_rel = os.path.relpath(peak, str(WEB_ROOT))
            p_rel = _url_path(p_rel)
            if ".." not in p_rel and "audit_frames" in p_rel:
                result["peak_image_url"] = f"/api/files/audit?p={p_rel}"
        except ValueError:
            pass
    if result.get("peak_image_path") and not result.get("peak_image_url"):
        result["peak_image_url"] = f"/api/files/audit?p=audit_frames/{os.path.basename(peak)}"
    return jsonify(result)


def _resolve_video_path(tool_id: str, video_id: int) -> str | None:
    if tool_id not in VIDEO_DIRS:
        return None
    files = _list_videos(VIDEO_DIRS[tool_id])
    if tool_id == "1" and not files:
        files = _list_videos("videos")
    if video_id < 0 or video_id >= len(files):
        return None
    return str(files[video_id])


@app.route("/api/run-overall", methods=["POST"])
def api_run_overall():
    data = request.get_json() or {}
    items = data.get("items") or []
    if len(items) < 2:
        return jsonify({"error": "至少需要 2 个视频（可来自不同工具）"}), 400
    resolved = []
    for it in items:
        tool_id = it.get("tool_id")
        video_id = it.get("video_id")
        if tool_id is None or video_id is None:
            return jsonify({"error": "每项需包含 tool_id 与 video_id"}), 400
        path = _resolve_video_path(tool_id, int(video_id))
        if not path:
            return jsonify({"error": f"无效的视频: tool_id={tool_id}, video_id={video_id}"}), 400
        resolved.append((tool_id, path))
    from .pipeline import run_overall_analysis
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_overall_analysis(resolved, progress_callback=None))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()
    return jsonify(result)


@app.route("/api/files/video")
def serve_video():
    p = request.args.get("p", "").lstrip("/")
    if ".." in p or not p.startswith("videos/"):
        return "Forbidden", 403
    path = WEB_ROOT / p
    if not path.is_file():
        return "Not Found", 404
    return send_from_directory(path.parent, path.name, mimetype="video/mp4")


@app.route("/api/files/audit")
def serve_audit():
    p = request.args.get("p", "").lstrip("/")
    if ".." in p or not p.startswith("audit_frames/"):
        return "Forbidden", 403
    path = WEB_ROOT / p
    if not path.is_file():
        return "Not Found", 404
    return send_from_directory(path.parent, path.name, mimetype="image/jpeg")


def main():
    from .pipeline import ensure_video_categories
    ensure_video_categories()
    print("AI 体操裁判 Web 服务启动中…")
    print("请打开 http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
