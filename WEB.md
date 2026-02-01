# AI 体操裁判 — 网页版

多页面网站流程：首页 → 选择分析工具 → 选择视频 → 运行分析（播放视频或显示峰值图）→ 简易动作报告 → 综合报告。界面为中文。

## 启动 Web 服务

**重要：请使用 Windows 自带的 PowerShell 或 CMD，不要用 Git Bash。** 在 Git Bash 下 `uv` 会报错（Unknown operating system: mingw_x86_64_msvcrt_gnu）。

在项目根目录任选一种方式：

**方式一（推荐）：用脚本**

```powershell
.\run_web.ps1
```

**方式二：先激活本项目的 .venv，再运行**

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.gymnastics_judge.web_app
```

**方式三：用 uv（在 PowerShell 下，且不要设置 VIRTUAL_ENV 到别的路径）**

```powershell
uv run python -m src.gymnastics_judge.web_app
```

若提示没有 `.venv`，先执行：`uv sync` 或 `pip install -e .`。

浏览器打开：**http://localhost:5000**

## 页面流程

1. **首页** — 标题与「开始分析」按钮  
2. **选择分析工具** — 三个按钮：平衡腿结环 (Penche)、交换腿鹿跳结环 (1.2096)、分腿跳结环 (1.2105)  
3. **选择视频** — 列出当前工具对应文件夹下的视频，选择后点击「开始分析」  
4. **分析结果** — 自动运行分析；Penche 显示可播放视频，其余两种显示生成的峰值图像；下方可进入简易报告与综合报告  
5. **简易动作报告** — 约 100 字：D 分、E 分、得分点/扣分点、一句话总结（参考 Questionnaire-AI 报告风格）  
6. **综合报告** — 约 200 字：做得好的地方、需改进处、建议下一步  

## 依赖

与 CLI 相同（需 `GOOGLE_API_KEY`、`pose_landmarker_full.task` 等）。另需 `flask`、`flask-cors`（已写入 `pyproject.toml`）。

## 静态文件

- `static/` — 所有页面与样式  
- `static/styles/common.css` — 通用样式（深色主题，参考 [Questionnaire-AI](https://github.com/wishang06/Questionnaire-AI) recruiter 风格）
