#!/usr/bin/env python3
"""analyze_frame_density.py — 分析板区文字密度分布，辅助标定 density 取帧阈值。

只读本地视频（ffmpeg 采样 + numpy 计算），不调用任何付费 API。

复用 extract_frames 的板区裁剪与密度采样实现（BOARD_CROP_GRAY +
_analyze_all_frames），输出:
  - 板区文字密度分位数与直方图（用于标定 floor / min_increment）
  - 空板区间估计（最低 1~5% 分位，可作为 floor 参考）

Usage:
    python3 analyze_frame_density.py --video output/xxx/xxx.mp4 [--sampling-fps 1.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def density_report(video: Path, fps: float, bright_threshold: int = 200) -> None:
    """采样板区文字密度，输出统计，用于标定 density 模式的 floor/increment。"""
    from extract_frames import BOARD_CROP_GRAY, _analyze_all_frames
    samples = _analyze_all_frames(video, fps, BOARD_CROP_GRAY, bright_threshold)
    n = len(samples)
    if n == 0:
        print("\n[density] 无采样帧")
        return
    dens = np.array([d for _, d, _ in samples])
    print(f"\n=== board 文字密度 (threshold>{bright_threshold}, "
          f"{n} 帧 @ {fps}fps) ===")
    pcts = np.percentile(dens, [1, 5, 25, 50, 75, 95, 99])
    print("密度分位数(%): " + ", ".join(
        f"{q}%={v:.3f}" for q, v in zip([1, 5, 25, 50, 75, 95, 99], pcts)))
    print(f"min={dens.min():.3f}  max={dens.max():.3f}  mean={dens.mean():.3f}")
    print(f"空板区间估计(最低1~5%分位): {pcts[0]:.3f}% ~ {pcts[1]:.3f}%")
    hist, edges = np.histogram(
        dens, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 3, 4, 6, 8, 10, 20, 100])
    for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
        if c:
            print(f"  [{lo:5.2f},{hi:5.2f})%: {c:5d} 帧  {'#' * min(c // 10, 60)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="板区文字密度分布分析（纯本地，标定 density 取帧阈值）")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--sampling-fps", type=float, default=1.0,
                    help="采样率 fps (默认 1.0)")
    ap.add_argument("--bright-threshold", type=int, default=200,
                    help="密度统计的亮像素阈值(0-255) (默认 200)")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] 视频不存在: {args.video}", file=sys.stderr)
        return 2

    density_report(args.video, args.sampling_fps, args.bright_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
