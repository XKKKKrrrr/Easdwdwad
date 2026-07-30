"""渲染源图与透视校正图并叠加碎片轮廓，供人工核验检测结果。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fragment_detector import detect_image, detect_poker_image


PALETTE = ((57, 201, 255), (79, 217, 135), (250, 174, 55), (220, 93, 215))


def _draw_label(image: np.ndarray, text: str, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    # 深色描边使编号在浅色碎片和蓝底上都保持可读。
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (18, 18, 18), 3, cv2.LINE_AA)
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _source_polygon(vertices_mm: list[list[float]], homography: np.ndarray, width_mm: float, height_mm: float) -> np.ndarray:
    """将毫米多边形先变为 1 px/mm 的校正坐标，再逆投影回源图。"""
    rectified = np.asarray(
        [[x + width_mm / 2.0, height_mm / 2.0 - z] for x, z in vertices_mm],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(rectified, np.linalg.inv(homography)).reshape(-1, 2).round().astype(np.int32)


def _render_cell(path: Path, detector, width: int = 760, height: int = 360) -> np.ndarray:
    """一个单元同时展示源图叠加和 1 px/mm 校正图叠加。"""
    result = detector(path)
    source = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    source_scale = min((width // 2 - 12) / source.shape[1], (height - 44) / source.shape[0])
    source_view = cv2.resize(source, None, fx=source_scale, fy=source_scale, interpolation=cv2.INTER_AREA)
    source_panel = np.full((height, width // 2, 3), 245, np.uint8)
    y = 28 + (height - 28 - source_view.shape[0]) // 2
    x = (source_panel.shape[1] - source_view.shape[1]) // 2
    source_panel[y : y + source_view.shape[0], x : x + source_view.shape[1]] = source_view
    _draw_label(source_panel, "source", (12, 21), (55, 55, 55))

    rectified_panel = np.full((height, width // 2, 3), 245, np.uint8)
    if result["ok"]:
        board = result["board"]
        corners = np.asarray(board["corners_px"], np.float32).reshape(-1, 1, 2)
        scale_matrix = np.array([[source_scale, 0, x], [0, source_scale, y], [0, 0, 1]], np.float32)
        cv2.polylines(source_panel, [cv2.perspectiveTransform(corners, scale_matrix).astype(np.int32)], True, (70, 220, 70), 2, cv2.LINE_AA)
        homography = np.asarray(board["homography"], dtype=np.float32)
        for index, piece in enumerate(result["pieces"]):
            color = PALETTE[index % len(PALETTE)]
            polygon = _source_polygon(piece["polygon_mm"], homography, board["width_mm"], board["height_mm"])
            polygon = (polygon.astype(np.float32) * source_scale + np.array([x, y])).round().astype(np.int32)
            cv2.polylines(source_panel, [polygon], True, color, 2, cv2.LINE_AA)

        # homography 按 1 px/mm 生成；预览必须沿用同一尺度，不能误用 2x 重试图。
        rectified = cv2.warpPerspective(source, homography, (round(board["width_mm"]), round(board["height_mm"])))
        rectified_scale = min((width // 2 - 12) / rectified.shape[1], (height - 44) / rectified.shape[0])
        rectified_view = cv2.resize(rectified, None, fx=rectified_scale, fy=rectified_scale, interpolation=cv2.INTER_AREA)
        ry = 28 + (height - 28 - rectified_view.shape[0]) // 2
        rx = (rectified_panel.shape[1] - rectified_view.shape[1]) // 2
        rectified_panel[ry : ry + rectified_view.shape[0], rx : rx + rectified_view.shape[1]] = rectified_view
        for index, piece in enumerate(result["pieces"]):
            color = PALETTE[index % len(PALETTE)]
            polygon = np.asarray(
                [[px + board["width_mm"] / 2.0, board["height_mm"] / 2.0 - pz] for px, pz in piece["polygon_mm"]],
                np.float32,
            )
            polygon = (polygon * rectified_scale + np.array([rx, ry])).round().astype(np.int32)
            cv2.polylines(rectified_panel, [polygon], True, color, 2, cv2.LINE_AA)
            _draw_label(rectified_panel, str(index + 1), tuple(polygon[0]), color)
        _draw_label(rectified_panel, f"rectified: {len(result['pieces'])} fragments", (12, 21), (55, 55, 55))
    else:
        _draw_label(rectified_panel, f"detection failed: {result['error']['code']}", (12, 21), (0, 0, 220))
    cell = np.hstack((source_panel, rectified_panel))
    cv2.putText(cell, path.name, (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    return cell


def render(dataset: Path, output: Path, poker: bool = False) -> None:
    """把数据集按两列拼为总览图；扑克开关仅切换检测前景模型。"""
    detector = detect_poker_image if poker else detect_image
    cells = [_render_cell(path, detector) for path in sorted(dataset.glob("*.jpg"))]
    if not cells:
        raise FileNotFoundError(f"No JPG files in {dataset}")
    rows = [np.hstack(cells[index : index + 2]) for index in range(0, len(cells), 2)]
    contact_sheet = np.vstack(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Unable to write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="包含待预览 JPG 图片的目录")
    parser.add_argument("--output", type=Path, default=Path("preview") / "fragment_detection_contact_sheet.jpg")
    parser.add_argument("--poker", action="store_true")
    args = parser.parse_args()
    render(args.dataset, args.output, args.poker)


if __name__ == "__main__":
    main()
