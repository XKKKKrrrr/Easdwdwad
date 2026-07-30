# 蓝色 A4 碎片识别

这是一个纯 Python 的单帧视觉识别工具，用于识别蓝色 A4 板面上的白色拼图碎片或扑克牌碎片。它只输出每片碎片的几何信息，不执行重排、矩形求解或机械臂控制。

## 特性

- 自动定位蓝色板面并进行透视校正。
- 白色碎片模式：HSV/Lab 颜色分割与 3 至 5 顶点多边形拟合。
- 扑克碎片模式：反选蓝色背景，使用凸包处理印刷花色造成的轮廓凹陷。
- 可尝试恢复轻微重叠的两个连通碎片。
- 输出毫米坐标、面积、主方向、置信度和板面单应矩阵。
- 仅依赖 NumPy 和 OpenCV；不依赖本机路径、C++ DLL、Shapely 或 SciPy。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 使用

```powershell
# 白色拼图碎片
python fragment_detector.py detect --input "image.jpg" --output "output\white.json"

# 扑克牌碎片
python fragment_detector.py detect --poker --input "image.jpg" --output "output\poker.json"

# 标注预览图，必须显式指定图片目录
python render_previews.py --dataset "images" --output "preview\contact_sheet.jpg"
python render_previews.py --dataset "poker_images" --poker --output "preview\poker_contact_sheet.jpg"
```

默认板面尺寸为 `206.6 x 293.0 mm`。使用标准 A4 图像时，可传入：

```powershell
python fragment_detector.py detect --input "image.jpg" --board-width-mm 210 --board-height-mm 297
```

## 输出

成功时会返回 JSON，包含 `board`、`pieces` 和 `diagnostics`。每个碎片的 `polygon_mm`、`centroid_mm` 与 `pickup_mm` 均为毫米坐标：纸张中心是原点，向右为 `+X`，向上为 `+Z`。

不会输出 `solution`，因为本项目不包含重排功能。

## 文件

- `fragment_detector.py`：白色/扑克碎片识别 CLI 与 Python API。
- `render_previews.py`：源图与透视校正图的轮廓预览生成器。
- `requirements.txt`：运行和测试依赖。

## 许可证

本目录未附带许可证。公开发布前，请由项目权利人选择并添加适用许可证。
