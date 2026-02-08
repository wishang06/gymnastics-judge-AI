# Video layout (categorized by element)

All videos live under **`videos/`** in this project, in subfolders by element:

| Folder           | Tool / element                               | Example filenames        |
|------------------|----------------------------------------------|--------------------------|
| `videos/penche/` | Penche (2.1106) — balance, hold, relevé      | `bad-1.mp4`, `Perfect-1.mp4` |
| `videos/1_2096/` | 交换腿鹿跳结环 (1.2096), FIG 9.31 | `12096_test_01.mp4`, `12096_test_02.mp4` |
| `videos/1_2105/` | 跨跳结环 (1.2105), FIG 9.29   | `12105_test_01.mp4`, `12105_test_02.mp4` |
| `videos/3_1203/` | 后屈腿转体 Back Attitude Pivot (3.1203), YOLO turn | `test_turn.mp4`, `test_turn1.mp4` |

- **Penche**: put existing penche `.mp4` files in `videos/penche/`. If that folder is empty, the app also looks in `videos/` for backward compatibility.
- **Rhythmic (1.2096 / 1.2105)**: filenames must match the element code. Put them in `videos/1_2096/` and `videos/1_2105/` respectively.
- **Turn (3.1203)**: Back Attitude Pivot uses YOLOv8-Pose. Put turn videos in `videos/3_1203/`. You must place **`yolov8x-pose.pt`** in the project root (copy from [Gymnastics_AI_Judge](https://github.com/ultralytics/ultralytics) or download; first run will download if using ultralytics).

## Moving dance_judge videos into this project

If your rhythmic videos are still in `dance_judge/my_dance_project/videos/`, copy them into this repo:

- Files like `12096_test_01.mp4`, `12096_test_02.mp4` → **`videos/1_2096/`**
- Files like `12105_test_01.mp4`, `12105_test_02.mp4` → **`videos/1_2105/`**

Example (PowerShell, run from project root):

```powershell
# Create category folders (already done by the app)
New-Item -ItemType Directory -Force -Path videos/penche, videos/1_2096, videos/1_2105, videos/3_1203

# Copy from dance_judge project (adjust path if needed)
$dance = "C:\Users\andys\Desktop\dance_judge\my_dance_project\videos"
Copy-Item "$dance\12096*.mp4" -Destination videos/1_2096/
Copy-Item "$dance\12105*.mp4" -Destination videos/1_2105/
```

After that, run the app and choose tool **2** or **3** to see the rhythmic videos listed.
