#!/usr/bin/env python3
"""preprocess_frames.py — 帧预处理：提取黑板/白板区域并清晰化。

在教学视频中，板书内容往往只占据画面的一部分（黑板/白板/屏幕）。
在送入大模型 OCR 之前，先裁剪出板面区域并做清晰化处理，
可显著提升 OCR 准确率。

处理流程（对每一帧）:
    1. 检测板面区域（白板或黑板），用行/列亮度投影定位板面矩形
    2. 裁剪到板面区域（可选，--no-crop 禁用）
    3. CLAHE 对比度增强 + 去噪 + 锐化

Usage:
    python3 preprocess_frames.py --manifest out/frames/frames.json --out-dir out/preprocessed
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def _find_band(profile: np.ndarray, threshold: float, mode: str,
               min_dark_ratio: float = 0.4) -> tuple[float, float] | None:
    """在 profile 中找板面范围，返回 (start_frac, end_frac)。

    mode='dark' 匹配低于阈值的暗区（黑板），mode='bright' 匹配高于阈值的亮区（白板）。
    取所有匹配区间的并集（允许中间有亮斑/阴影造成的断续），
    但要求匹配区占比 >= min_dark_ratio，否则视为未检测到。
    """
    n = len(profile)
    hit = profile < threshold if mode == "dark" else profile > threshold
    if float(np.mean(hit)) < min_dark_ratio:
        return None
    idx = np.where(hit)[0]
    return float(idx.min()) / n, float(idx.max()) / n


def detect_board(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """检测板面区域，返回 (x, y, w, h)。

    用行/列亮度投影找暗区（黑板）或亮区（白板）：
      - 垂直方向: 黑板/白板在教学视频中通常位于画面中部，上下是
        天花板/讲台等亮区，投影后形成明显的高对比区域。
      - 水平方向: 在垂直带内再找水平范围，取暗列并集剔除两侧干扰。
    相比"最大轮廓"法，投影法不受教师深色衣服/讲台阴影影响，更稳健。
    """
    h, w = gray.shape
    mean = float(np.mean(gray))
    is_whiteboard = mean > 127
    mode = "bright" if is_whiteboard else "dark"

    # OTSU 自适应阈值
    t_otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 垂直投影：每行亮度 → 找板面垂直范围
    row_profile = np.interp(
        np.linspace(0, h - 1, 100), np.arange(h), gray.mean(axis=1))
    band_y = _find_band(row_profile, float(t_otsu), mode)
    if band_y is None:
        return None
    y0, y1 = int(band_y[0] * h), int(band_y[1] * h)

    # 水平投影：在垂直范围内每列亮度 → 找板面水平范围
    col_profile = np.interp(
        np.linspace(0, w - 1, 100), np.arange(w), gray[y0:y1].mean(axis=0))
    band_x = _find_band(col_profile, float(t_otsu), mode)
    x0, x1 = (int(band_x[0] * w), int(band_x[1] * w)) if band_x else (0, w)

    bw, bh = x1 - x0, y1 - y0
    area_ratio = (bw * bh) / (h * w)
    if area_ratio < 0.15:
        return None

    # 放宽边缘（板面通常略小于检测框）
    pad_x, pad_y = int(bw * 0.02), int(bh * 0.02)
    x = max(0, x0 - pad_x)
    y = max(0, y0 - pad_y)
    bw = min(w - x, bw + 2 * pad_x)
    bh = min(h - y, bh + 2 * pad_y)
    return x, y, bw, bh


def enhance(img: np.ndarray) -> np.ndarray:
    """清晰化：CLAHE 对比度增强 + 降噪 + 锐化。"""
    # 1. 自适应对比度增强 (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 2. 降噪（保边）
    img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

    # 3. 锐化（unsharp mask）
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return img


def process_frame(src: Path, dst: Path, do_crop: bool) -> tuple[int, int, int, int] | None:
    """处理单帧：裁剪 + 增强。返回板面框 (x,y,w,h) 或 None。"""
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图片: {src}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = detect_board(gray) if do_crop else None
    if box:
        x, y, w, h = box
        img = img[y : y + h, x : x + w]

    img = enhance(img)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return box


def main() -> int:
    ap = argparse.ArgumentParser(description="帧预处理：板面提取 + 清晰化")
    ap.add_argument("--manifest", required=True, type=Path,
                    help="extract_frames.py 输出的 frames.json")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="预处理后帧的输出目录")
    ap.add_argument("--no-crop", action="store_true",
                    help="不裁剪板面，只做增强")
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"[err] manifest 不存在: {args.manifest}", file=sys.stderr)
        return 2

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        print("[err] manifest 中没有帧记录", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pre] {len(frames)} 帧预处理 (crop={'off' if args.no_crop else 'on'})...",
          file=sys.stderr)

    crop_stats = []
    new_frames = []
    for i, fr in enumerate(frames):
        src = Path(fr["file"])
        dst = out_dir / f"frame_{i + 1:06d}.jpg"
        try:
            box = process_frame(src, dst, do_crop=not args.no_crop)
        except Exception as e:
            print(f"[err] 帧 {i + 1} 处理失败: {e}", file=sys.stderr)
            continue
        if box:
            crop_stats.append(1)
        new_frames.append({"file": str(dst), "t": fr["t"]})

    # 覆写 manifest 指向预处理后的帧
    data["frames"] = new_frames
    data["preprocessed"] = True
    data["cropped"] = sum(crop_stats)
    data["total"] = len(new_frames)
    manifest_out = out_dir / "frames.json"
    manifest_out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] 处理 {len(new_frames)} 帧, 其中 {len(crop_stats)} 帧检测到板面 "
          f"-> {out_dir}", file=sys.stderr)
    print(f"[ok] manifest -> {manifest_out}")
    print(str(manifest_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
