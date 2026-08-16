#!/usr/bin/env python3
"""analyze_frame_dedup.py — 分析视频帧间 dHash 距离分布，辅助确定 dedup 阈值。

只读本地视频（ffmpeg 采样 + numpy 计算），不调用任何付费 API。

输出:
  - 指定采样率下相邻帧对的 dHash 汉明距离分布（分位数、直方图）
  - 不同阈值下 dedup 保留帧数（含区域裁剪对比: full / top / center）
  - 实际板书块数量对比（用于判断阈值是否能把 21 块板书切分出来）

Usage:
    python3 analyze_frame_dedup.py --video output/xxx/xxx.mp4 [--fps 1.0]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# 导入 config 以把项目 bin/ 目录加入 PATH（ffmpeg）
from config import BIN_DIR  # noqa: F401  (副作用: 注册 bin 到 PATH)

# 区域裁剪 -> ffmpeg 滤镜链（crop 后统一缩放到 9x8 灰度，保持与 extract_frames 一致）
REGION_FILTERS = {
    "full":   "scale=9:8",
    "top":    "crop=iw:ih*0.5:0:0,scale=9:8",
    "center": "crop=iw:ih*0.6:0:ih*0.2,scale=9:8",
    # 板书区域：画面 15%~60% 高度，覆盖黑板中上部，避开底部高动态的教师身体/讲台
    "board":  "crop=iw:ih*0.45:0:ih*0.15,scale=9:8",
}


def sample_hashes(video: Path, fps: float, region: str) -> list[np.ndarray]:
    """以 fps 采样视频，返回每帧的 64-bit dHash（np.bool_ 数组）。"""
    vf = f"fps={fps},{REGION_FILTERS[region]},format=gray"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", vf, "-f", "image2pipe", "-vcodec", "pgm", "-"],
        capture_output=True, check=True,
    )
    out = []
    for body in proc.stdout.split(b"P5\n")[1:]:
        try:
            nl1 = body.index(b"\n")
            nl2 = body.index(b"\n", nl1 + 1)
            w, h = (int(x) for x in body[:nl1].split())
            px = np.frombuffer(body[nl2 + 1:nl2 + 1 + w * h],
                               dtype=np.uint8).reshape(h, w)
            out.append(px[:, 1:] > px[:, :-1])
        except Exception:
            break
    return out


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def kept_count(hashes: list[np.ndarray], threshold: int) -> int:
    """模拟 dedup 保留逻辑：与上一保留帧距离 > threshold 则保留。"""
    if not hashes:
        return 0
    kept = 1
    last = hashes[0]
    for h in hashes[1:]:
        if hamming(h, last) > threshold:
            kept += 1
            last = h
    return kept


def kept_idxs(hashes: list[np.ndarray], threshold: int) -> list[int]:
    """模拟 dedup 保留逻辑，返回保留帧索引。"""
    if not hashes:
        return []
    idxs = [0]
    last = hashes[0]
    for i, h in enumerate(hashes[1:], 1):
        if hamming(h, last) > threshold:
            idxs.append(i)
            last = h
    return idxs


def boundary_hits(hashes: list[np.ndarray], boundaries: list[float],
                  fps: float, win: float = 5.0) -> None:
    """统计各阈值下保留帧对板书边界时间的命中情况（供选阈值）。"""
    if not boundaries:
        return
    idx_bounds = [int(round(b * fps)) for b in boundaries]
    print(f"板书边界 {len(boundaries)} 个 @ {[f'{b:.0f}s' for b in boundaries]}")
    print("阈值 | 保留帧 | 命中边界(±%ds) | 噪声帧(离任何边界>%ds) | 覆盖块数" % (win, win))
    for t in [6, 8, 10, 12, 15, 18, 20, 25]:
        idxs = kept_idxs(hashes, t)
        hit = set()
        noise = 0
        for i in idxs:
            s = i / fps
            hit_b = [b for b in idx_bounds if abs(s - b / fps) <= win]
            if hit_b:
                hit.update(hit_b)
            else:
                noise += 1
        print(f"  >{t:2d} | {len(idxs):4d}   | {len(hit):4d}          | {noise:5d}              | {len(hit)}")


def report(name: str, hashes: list[np.ndarray], boundaries: list[float],
           fps: float) -> None:
    n = len(hashes)
    if n < 2:
        print(f"{name}: 帧数不足 ({n})")
        return
    dists = np.array([hamming(a, b) for a, b in zip(hashes, hashes[1:])])
    pcts = np.percentile(dists, [10, 25, 50, 75, 90, 95, 99])
    print(f"\n=== {name} (采样帧数 {n}, 相邻帧对数 {len(dists)}) ===")
    print("相邻帧距离分位数: "
          + ", ".join(f"{q}%={int(v)}" for q, v in zip([10, 25, 50, 75, 90, 95, 99], pcts)))
    print("均值=%.2f 标准差=%.2f 最小值=%d 最大值=%d"
          % (dists.mean(), dists.std(), dists.min(), dists.max()))
    # 直方图（每 4 位一档）
    hist, edges = np.histogram(dists, bins=range(0, 68, 4))
    for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
        if c:
            print(f"  [{lo:2d},{hi:2d}) 距离: {c:4d} 帧对  {'#' * min(c // 5, 60)}")
    # 阈值扫描
    print("不同阈值下的 dedup 保留帧数:")
    for t in [4, 6, 8, 10, 12, 15, 20, 25, 30]:
        print(f"  hamming>{t}: {kept_count(hashes, t):4d} 帧")
    # 保留帧时间分布（阈值 10 和 15），观察是否集中或均匀
    for t in [10, 15]:
        idxs = kept_idxs(hashes, t)
        if len(idxs) < 2:
            print(f"  [阈值>{t}] 保留 {len(idxs)} 帧（不足两帧，无间隔统计）")
            continue
        gaps = np.diff(idxs)
        print(f"  [阈值>{t}] 保留 {len(idxs)} 帧, 帧间隔分位: "
              f"min={gaps.min()} p50={int(np.median(gaps))} p90={int(np.percentile(gaps,90))} "
              f"max={gaps.max()}")

    if boundaries:
        boundary_hits(hashes, boundaries, fps)


def main() -> int:
    ap = argparse.ArgumentParser(description="dHash 帧间距离分布分析（纯本地）")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=1.0,
                    help="采样率 fps (默认 1.0)")
    ap.add_argument("--region", choices=["all", "full", "top", "center", "board"],
                    default="all", help="分析区域 (默认 all)")
    ap.add_argument("--captions", type=Path, default=None,
                    help="captions.json（板书 start 列表），用于统计边界命中率")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] 视频不存在: {args.video}", file=sys.stderr)
        return 2

    boundaries: list[float] = []
    if args.captions and args.captions.is_file():
        import json
        data = json.loads(args.captions.read_text(encoding="utf-8"))
        caps = data if isinstance(data, list) else (
            data.get("captions") or data.get("segments") or [])
        boundaries = [float(c["start"]) for c in caps if c.get("text")]
        print(f"[info] 从 {args.captions.name} 读取 {len(boundaries)} 个板书边界\n")

    regions = ["full", "top", "center"] if args.region == "all" else [args.region]
    for r in regions:
        hashes = sample_hashes(args.video, args.fps, r)
        report(r, hashes, boundaries, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
