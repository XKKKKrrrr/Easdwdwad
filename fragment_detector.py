"""蓝色 A4 板面上的碎片检测，不执行拼图重排。"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


DEFAULT_BOARD_WIDTH_MM = 206.6
DEFAULT_BOARD_HEIGHT_MM = 293.0
# 赛题输入固定为 2 到 4 片；该范围同时限制遮挡切分的误检扩张。
MIN_PIECES = 2
MAX_PIECES = 4
ERROR_CODES = {"IMAGE_INVALID", "IMAGE_BLURRED", "BOARD_NOT_FOUND", "PIECE_COUNT_INVALID", "POLYGON_INVALID"}


class VisionError(RuntimeError):
    def __init__(self, code: str, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.diagnostics = diagnostics or {}


@dataclass
class Board:
    # geometry 用于 1 px/mm 首轮检测；texture 为 2 px/mm 重试保留原始细节。
    source: np.ndarray
    corners_px: np.ndarray
    homography: np.ndarray
    geometry: np.ndarray
    texture: np.ndarray
    width_mm: float
    height_mm: float
    geometry_scale: float = 1.0
    texture_scale: float = 2.0
    high_resolution_retry: bool = False
    input_overlap_detected: bool = False
    recovered_occluded_pieces: int = 0


@dataclass
class Piece:
    # vertices、centroid、pickup 均使用以 A4 中心为原点的 (X, Z) 毫米坐标。
    vertices: np.ndarray
    area: float
    centroid: np.ndarray
    pickup: np.ndarray
    angle: float
    confidence: float


def _point_json(point: np.ndarray, digits: int = 3) -> list[float]:
    return [round(float(point[0]), digits), round(float(point[1]), digits)]


def _polygon_json(vertices: np.ndarray) -> list[list[float]]:
    return [_point_json(point) for point in vertices]


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _polygon_area(vertices: np.ndarray) -> float:
    """用鞋带公式计算有向面积，符号用于统一顶点绕序。"""
    x, y = vertices[:, 0], vertices[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """按多边形面积计算质心；退化多边形才回退为顶点均值。"""
    area = _polygon_area(vertices)
    if abs(area) < 1e-9:
        return vertices.mean(axis=0)
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    return np.array([
        np.sum((vertices[:, 0] + following[:, 0]) * cross) / (6.0 * area),
        np.sum((vertices[:, 1] + following[:, 1]) * cross) / (6.0 * area),
    ])


def _order_quad(points: np.ndarray) -> np.ndarray:
    """按左上、右上、右下、左下排序，供透视变换稳定使用。"""
    if len(points) != 4:
        raise VisionError("BOARD_NOT_FOUND", "Board quadrilateral does not have four corners")
    sums = points.sum(axis=1)
    diffs = points[:, 1] - points[:, 0]
    return np.array([points[np.argmin(sums)], points[np.argmin(diffs)], points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32)


def _detect_board_quad(source: np.ndarray) -> np.ndarray:
    """在缩小图中提取最大蓝色区域，再回投到原图坐标。"""
    scale = min(1.0, 640.0 / max(source.shape[:2]))
    small = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # 与嵌入版一致的宽松蓝色范围，先保证整张板不会被光照边缘截断。
    blue = ((hue >= 82) & (hue <= 142) & (saturation >= 55) & (value >= 35)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise VisionError("BOARD_NOT_FOUND", "No blue board region was detected")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) / float(small.shape[0] * small.shape[1]) < 0.12:
        raise VisionError("BOARD_NOT_FOUND", "Blue board region is too small")
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    approximation = None
    for factor in (0.012, 0.018, 0.025, 0.035, 0.05):
        candidate = cv2.approxPolyDP(hull, factor * perimeter, True)
        if len(candidate) == 4:
            approximation = candidate[:, 0, :]
            break
    # 无法可靠化简为四边形时，使用最小外接矩形保持可用的板面坐标系。
    if approximation is None or not cv2.isContourConvex(approximation):
        approximation = cv2.boxPoints(cv2.minAreaRect(hull))
    ordered = _order_quad(np.asarray(approximation, dtype=np.float32) / scale)
    first_axis = np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])
    second_axis = np.linalg.norm(ordered[2] - ordered[1]) + np.linalg.norm(ordered[3] - ordered[0])
    return np.roll(ordered, -1, axis=0) if first_axis > second_axis else ordered


def _warp_board(source: np.ndarray, corners: np.ndarray, width_mm: float, height_mm: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """将源图校正到指定像素密度的板面，并返回源图到板面的单应矩阵。"""
    width, height = max(1, round(width_mm * scale)), max(1, round(height_mm * scale))
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(source, homography, (width, height), flags=cv2.INTER_LINEAR), homography


def _prepare_board(source: np.ndarray, width_mm: float, height_mm: float, strict_quality: bool) -> Board:
    """进行输入尺寸、清晰度、板面尺寸校验并预生成两种检测分辨率。"""
    if source.ndim != 3 or source.shape[0] < 240 or source.shape[1] < 320:
        raise VisionError("IMAGE_INVALID", "Input image is empty or too small")
    if not 180.0 <= width_mm <= 220.0 or not 270.0 <= height_mm <= 310.0:
        raise VisionError("IMAGE_INVALID", "Board dimensions are outside the supported range")
    if strict_quality:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < 18.0:
            raise VisionError("IMAGE_BLURRED", "Input image is too blurred", {"laplacian_variance": blur_score})
    corners = _detect_board_quad(source)
    geometry, homography = _warp_board(source, corners, width_mm, height_mm, 1.0)
    texture, _ = _warp_board(source, corners, width_mm, height_mm, 2.0)
    return Board(source, corners, homography, geometry, texture, width_mm, height_mm)


def _segment(board_image: np.ndarray, pixels_per_mm: float, poker: bool = False) -> np.ndarray:
    """生成已填充的候选连通域掩膜，白色和扑克模式仅在前景定义上不同。"""
    hsv = cv2.cvtColor(board_image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if poker:
        # 扑克包含红黑花色，不能按白色阈值分割；直接反选蓝色背景保留整张牌。
        blue = (hue >= 82) & (hue <= 142) & (saturation >= 55) & (value >= 35)
        mask = (~blue).astype(np.uint8) * 255
    else:
        # 白色碎片要求低饱和、足够亮且 Lab 色度接近中性，抑制蓝底反光。
        lab = cv2.cvtColor(board_image, cv2.COLOR_BGR2LAB)
        _, lab_a, lab_b = cv2.split(lab)
        mask = ((saturation < 110) & (value > 105) & (np.abs(lab_a.astype(np.int16) - 128) < 28) & (np.abs(lab_b.astype(np.int16) - 128) < 28)).astype(np.uint8) * 255
    margin = max(2, round(2.0 * pixels_per_mm))
    mask[:margin] = mask[-margin:] = 0
    mask[:, :margin] = mask[:, -margin:] = 0
    mask = cv2.medianBlur(mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    minimum_area = (120.0 if poker else 80.0) * pixels_per_mm * pixels_per_mm
    # 先填充合格大区域，避免文字、边缘噪点在后续轮廓提取中形成假碎片。
    for contour in contours:
        if cv2.contourArea(contour) >= minimum_area:
            cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    return filled


def _fit_piece(contour: np.ndarray, pixels_per_mm: float, width_mm: float, height_mm: float, poker: bool = False) -> Piece | None:
    """按嵌入版 epsilon 扫描，把候选轮廓化简为 3 至 5 边碎片。"""
    # 扑克的印刷和轻微遮挡会产生凹口；先取凸包才能稳定还原其外轮廓。
    fitting_contour = cv2.convexHull(contour) if poker else contour
    contour_area = abs(cv2.contourArea(fitting_contour))
    best_score, best_error, best = float("inf"), 1.0, None
    # 主搜索从 0.35 到 3.55 mm，优先顶点少、面积误差小且没有过短边的多边形。
    for step in range(81):
        candidate = cv2.approxPolyDP(fitting_contour, (0.35 + step * 0.04) * pixels_per_mm, True)[:, 0, :]
        if not 3 <= len(candidate) <= 5 or not cv2.isContourConvex(candidate):
            continue
        lengths = np.linalg.norm(np.roll(candidate, -1, axis=0).astype(float) - candidate.astype(float), axis=1) / pixels_per_mm
        area_error = abs(abs(cv2.contourArea(candidate)) - contour_area) / max(1.0, contour_area)
        if lengths.min() < 7.5 or area_error > 0.16:
            continue
        score = len(candidate) + area_error * 1.5 + max(0.0, (20.0 - float(lengths.min())) / 20.0) * 0.08
        if score < best_score:
            best_score, best_error, best = score, area_error, candidate
    if best is None and poker:
        # 扑克碎片允许更小的花色附近边缘，按原生实现放宽门槛但降低置信度。
        for step in range(121):
            candidate = cv2.approxPolyDP(fitting_contour, (0.35 + step * 0.04) * pixels_per_mm, True)[:, 0, :]
            if not 3 <= len(candidate) <= 5 or not cv2.isContourConvex(candidate):
                continue
            lengths = np.linalg.norm(np.roll(candidate, -1, axis=0).astype(float) - candidate.astype(float), axis=1) / pixels_per_mm
            area_error = abs(abs(cv2.contourArea(candidate)) - contour_area) / max(1.0, contour_area)
            if lengths.min() < 3.0 or area_error > 0.18:
                continue
            score = len(candidate) + area_error * 1.5 + max(0.0, (7.5 - float(lengths.min())) / 7.5) * 0.2
            if score < best_score:
                best_score, best_error, best = score, area_error + 0.08, candidate
    if best is None:
        return None
    vertices = np.column_stack((best[:, 0] / pixels_per_mm - width_mm / 2.0, height_mm / 2.0 - best[:, 1] / pixels_per_mm)).astype(float)
    # 统一为逆时针绕序，保证面积、角度和下游消费端无须处理双重约定。
    if _polygon_area(vertices) < 0:
        vertices = vertices[::-1]
    edges = np.roll(vertices, -1, axis=0) - vertices
    longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    angle = math.atan2(float(longest[1]), float(longest[0]))
    if angle < -math.pi / 2 or angle >= math.pi / 2:
        angle = _normalize_angle(angle + math.pi)
    centroid = _polygon_centroid(vertices)
    return Piece(vertices, abs(_polygon_area(vertices)), centroid, centroid, angle, max(0.0, min(1.0, 1.0 - best_error * 4.0)))


def _split_occluded_contour(contour: np.ndarray, image: np.ndarray, pixels_per_mm: float) -> list[np.ndarray]:
    """利用凸缺陷和内部梯度，把轻微重叠形成的连通域尝试拆成两片。"""
    if len(contour) < 12 or cv2.contourArea(contour) < 400.0 * pixels_per_mm * pixels_per_mm:
        return []
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    hull_points = cv2.convexHull(contour)
    # 接近凸形的普通碎片不拆分，避免把 JPEG 锯齿误当成重叠边界。
    if hull_indices is None or len(hull_indices) < 4 or cv2.contourArea(contour) / max(1.0, cv2.contourArea(hull_points)) > 0.90:
        return []
    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return []
    defect_rows = np.asarray(defects).reshape(-1, 4)
    concave = [contour[row[2]][0] for row in defect_rows if row[3] / (256.0 * pixels_per_mm) >= 1.5]
    if not 2 <= len(concave) <= 8:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    component = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)
    best_score, best = -1.0, []
    # 穷举凹点连线，仅接受具有足够 Sobel 边缘支持且能切成两个大区域的方案。
    for first in range(len(concave)):
        for second in range(first + 1, len(concave)):
            a, b = tuple(map(int, concave[first])), tuple(map(int, concave[second]))
            if not 8.0 <= np.linalg.norm(np.asarray(a) - np.asarray(b)) / pixels_per_mm <= 100.0:
                continue
            samples = np.linspace(a, b, 8).round().astype(int)
            edge_support = float(np.mean(gradient[samples[:, 1], samples[:, 0]]))
            if edge_support < 18.0:
                continue
            cut = component.copy()
            cv2.line(cut, a, b, 0, max(2, round(pixels_per_mm)), cv2.LINE_8)
            parts, _ = cv2.findContours(cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            parts = [part for part in parts if cv2.contourArea(part) >= 120.0 * pixels_per_mm * pixels_per_mm]
            if len(parts) != 2:
                continue
            score = edge_support + min(cv2.contourArea(parts[0]), cv2.contourArea(parts[1])) / max(1.0, cv2.contourArea(contour)) * 80.0
            if score > best_score:
                best_score, best = score, parts
    return best


def _detect_at_scale(board: Board, pixels_per_mm: float, poker: bool = False) -> list[Piece]:
    """在单一像素密度下完成分割、拟合、可选遮挡恢复和稳定排序。"""
    image = board.geometry if pixels_per_mm == board.geometry_scale else board.texture
    mask = _segment(image, pixels_per_mm, poker)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    direct = [_fit_piece(contour, pixels_per_mm, board.width_mm, board.height_mm, poker) for contour in contours]
    direct_count = sum(piece is not None for piece in direct)
    pieces: list[Piece] = []
    for contour, fitted in zip(contours, direct):
        split = _split_occluded_contour(contour, image, pixels_per_mm) if direct_count < MAX_PIECES else []
        recovered = [_fit_piece(part, pixels_per_mm, board.width_mm, board.height_mm, poker) for part in split]
        if split and all(piece is not None for piece in recovered) and direct_count - int(fitted is not None) + len(recovered) <= MAX_PIECES:
            board.input_overlap_detected = True
            board.recovered_occluded_pieces += len(recovered)
            pieces.extend(piece for piece in recovered if piece is not None)
            direct_count += len(recovered) - int(fitted is not None)
        elif fitted is not None:
            pieces.append(fitted)
    if not MIN_PIECES <= len(pieces) <= MAX_PIECES:
        raise VisionError("PIECE_COUNT_INVALID", f"Expected 2-4 complete fragments, detected {len(pieces)}", {"detected_count": len(pieces), "scale": pixels_per_mm})
    return sorted(pieces, key=lambda piece: (-piece.centroid[1], piece.centroid[0]))


def _detect_pieces(board: Board, poker: bool = False) -> list[Piece]:
    """先快速检测；失败或任一碎片低置信度时强制使用 2 px/mm 重试。"""
    try:
        pieces = _detect_at_scale(board, board.geometry_scale, poker)
        if all(piece.confidence >= 0.82 for piece in pieces):
            return pieces
    except VisionError:
        pass
    board.high_resolution_retry = True
    board.input_overlap_detected = False
    board.recovered_occluded_pieces = 0
    pieces = _detect_at_scale(board, board.texture_scale, poker)
    board.geometry = board.texture
    board.geometry_scale = board.texture_scale
    return pieces


def _success(source: str, board: Board, pieces: list[Piece], elapsed_ms: float, poker: bool = False) -> dict[str, Any]:
    """按原 Python 检测接口组织 JSON，刻意不生成 solution 字段。"""
    return {
        "version": "2.0",
        "ok": True,
        "source": source,
        "elapsed_ms": round(elapsed_ms, 3),
        "board": {"corners_px": [_point_json(point, 2) for point in board.corners_px], "homography": [[round(float(value), 8) for value in row] for row in board.homography], "width_mm": board.width_mm, "height_mm": board.height_mm, "rotated_180": False},
        "pieces": [
            {"id": f"piece-{index}", "polygon_mm": _polygon_json(piece.vertices), "centroid_mm": _point_json(piece.centroid), "angle_rad": round(piece.angle, 4), "pickup_mm": _point_json(piece.pickup), "vertex_count": len(piece.vertices), "area_mm2": round(piece.area, 4), "confidence": round(piece.confidence, 4), "contour_source": "card" if poker else "paper"}
            for index, piece in enumerate(pieces, start=1)
        ],
        "diagnostics": {"backend": "python-opencv", "geometry_scale_px_per_mm": board.geometry_scale, "texture_scale_px_per_mm": board.texture_scale, "high_resolution_retry": board.high_resolution_retry, "input_overlap_detected": board.input_overlap_detected, "recovered_occluded_pieces": board.recovered_occluded_pieces},
    }


def detect_jpeg(image_bytes: bytes, *, source: str = "<encoded-image>", strict_quality: bool = True, board_width_mm: float = DEFAULT_BOARD_WIDTH_MM, board_height_mm: float = DEFAULT_BOARD_HEIGHT_MM, poker: bool = False) -> dict[str, Any]:
    """统一的字节入口，所有视觉错误都转换为稳定的 JSON 错误对象。"""
    started = time.perf_counter()
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise VisionError("IMAGE_INVALID", "Unable to decode JPG or PNG bytes")
        board = _prepare_board(image, board_width_mm, board_height_mm, strict_quality)
        return _success(source, board, _detect_pieces(board, poker), (time.perf_counter() - started) * 1000.0, poker)
    except VisionError as error:
        return {"version": "2.0", "ok": False, "source": source, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3), "error": {"code": error.code, "message": error.message}, "diagnostics": error.diagnostics}


def detect_image(source_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """白色碎片的文件入口。"""
    path = Path(source_path)
    if not path.is_file():
        return {"version": "2.0", "ok": False, "source": str(path), "elapsed_ms": 0.0, "error": {"code": "IMAGE_INVALID", "message": f"Image does not exist: {path}"}, "diagnostics": {}}
    return detect_jpeg(path.read_bytes(), source=str(path), **kwargs)


def detect_poker_image(source_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """扑克碎片的文件入口；与白色碎片共享板面定位和几何拟合管线。"""
    return detect_image(source_path, poker=True, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detect", nargs="?")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--allow-low-quality", action="store_true")
    parser.add_argument("--poker", action="store_true", help="使用扑克碎片的反蓝底分割与凸包拟合")
    parser.add_argument("--board-width-mm", type=float, default=DEFAULT_BOARD_WIDTH_MM)
    parser.add_argument("--board-height-mm", type=float, default=DEFAULT_BOARD_HEIGHT_MM)
    args = parser.parse_args(argv)
    result = detect_image(
        args.input,
        strict_quality=not args.allow_low_quality,
        board_width_mm=args.board_width_mm,
        board_height_mm=args.board_height_mm,
        poker=args.poker,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
