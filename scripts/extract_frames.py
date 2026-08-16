#!/usr/bin/env python3
"""extract_frames.py — 从视频中采样代表性帧。

两种模式:
  - interval (可选): 均匀 fps 采样
  - dedup (默认): 密集采样 + 感知哈希(dHash)去重，适合板书/录屏类视频

dedup 模式调优说明（依据 analyze_frame_dedup.py 对真实教学视频的测量）:
  - 采样率 1fps：21 分钟视频约 1289 帧，开销可忽略。
  - 区域裁剪: 默认只对画面 15%~60% 高度（板书区域）计算 dHash。
    实测该区域相邻帧距离 p95≈9，而画面底部（教师身体/讲台）时变方差是板书区
    的数倍，全帧哈希会把教师走动误判为内容变化；只统计板书区可显著降噪。
  - 汉明阈值: 默认 10，与该区域相邻帧距离 p95 对齐；实测保留约 6% 采样帧，
    远低于 max_frames 上限。
  - max_frames: 默认 120，作为安全上限（每帧对应一次 OCR 计费）。

输出: <out_dir>/frame_%06d.jpg + frames.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# 导入 config 以把项目 bin/ 目录加入 PATH（ffmpeg/ffprobe）
from config import BIN_DIR  # noqa: F401  (副作用: 注册 bin 到 PATH)

# dHash 计算区域 -> ffmpeg 滤镜链（先裁剪再缩放到 9x8 灰度）。
# 教学视频中板书通常位于画面中上部，画面底部多为教师身体/讲台等
# 高动态干扰；默认只统计 "board" 区域以降低人物走动导致的误判。
REGION_FILTERS = {
    "full":   "scale=9:8",
    "top":    "crop=iw:ih*0.5:0:0,scale=9:8",
    "center": "crop=iw:ih*0.6:0:ih*0.2,scale=9:8",
    "board":  "crop=iw:ih*0.45:0:ih*0.15,scale=9:8",
}


def ffprobe_duration(video: Path) -> float:
    """获取视频时长（秒）。"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def _dhash_bits(pix: np.ndarray) -> np.ndarray:
    """9x8 灰度图 → 64-bit 布尔数组（相邻像素比较）。"""
    return pix[:, 1:] > pix[:, :-1]


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def extract_interval(video: Path, out_dir: Path, interval: float,
                     fps_filter: float | None) -> list[dict]:
    """均匀 fps 采样。时间戳 = 等间距分布在视频时长上。"""
    filt = f"fps={fps_filter}" if fps_filter else f"fps={1.0 / interval}"
    pattern = str(out_dir / "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", filt, "-q:v", "3", "-y", pattern],
        check=True,
    )
    frames = sorted(out_dir.glob("frame_*.jpg"))
    duration = ffprobe_duration(video)
    n = len(frames)
    records = []
    for i, f in enumerate(frames):
        t = round(i * (duration / n), 3) if n else 0.0
        records.append({"file": str(f), "t": t})
    return records


def _hash_all_frames(video: Path, fps: float, region: str) -> list[tuple[int, np.ndarray]]:
    """对视频以 fps 采样，返回 [(帧索引, dhash)] 列表。

    通过 ffmpeg 管道输出 PGM 流，逐帧解析计算哈希。
    region 指定参与哈希的画面区域（见 REGION_FILTERS）。
    """
    vf = f"fps={fps},{REGION_FILTERS[region]},format=gray"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", vf, "-f", "image2pipe",
         "-vcodec", "pgm", "-"],
        capture_output=True, check=True,
    )
    blob = proc.stdout
    out = []
    idx = 0
    chunks = blob.split(b"P5\n")
    for body in chunks[1:]:
        try:
            nl1 = body.index(b"\n")
            nl2 = body.index(b"\n", nl1 + 1)
            w, h = (int(x) for x in body[:nl1].split())
            px = np.frombuffer(body[nl2 + 1:nl2 + 1 + w * h],
                               dtype=np.uint8).reshape(h, w)
            out.append((idx, _dhash_bits(px)))
            idx += 1
        except Exception:
            break
    return out


def extract_dedup(video: Path, out_dir: Path, fps: float, hamming: int,
                  max_frames: int, region: str) -> list[dict]:
    """密集采样 + dHash 去重 + 帧数上限裁剪。

    保留与上一保留帧汉明距离 > hamming 的帧，最后均匀降采样到 max_frames。
    """
    all_hashes = _hash_all_frames(video, fps, region)
    if not all_hashes:
        return []

    def t_of(i: int) -> float:
        return round(i / fps, 3)

    kept_idx = [all_hashes[0][0]]
    last = all_hashes[0][1]
    for i, h in all_hashes[1:]:
        if _hamming(h, last) > hamming:
            kept_idx.append(i)
            last = h

    if len(kept_idx) > max_frames:
        step = len(kept_idx) / max_frames
        kept_idx = [kept_idx[int(j * step)] for j in range(max_frames)]

    records = []
    for n, fi in enumerate(kept_idx, 1):
        jpg = out_dir / f"frame_{n:06d}.jpg"
        t = t_of(fi)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", str(t), "-i", str(video), "-frames:v", "1",
             "-q:v", "3", "-y", str(jpg)],
            check=True,
        )
        if jpg.exists():
            records.append({"file": str(jpg), "t": t})
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="视频帧采样工具")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["interval", "dedup"], default="dedup",
                    help="帧采样模式 (默认 dedup)")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="interval 模式的帧间隔秒数 (默认 2.0)")
    ap.add_argument("--fps", type=float, default=None,
                    help="显式 fps 滤镜值 (覆盖 --interval)")
    ap.add_argument("--dedup-fps", type=float, default=1.0,
                    help="dedup 模式密集采样率 (默认 1.0 fps)")
    ap.add_argument("--dedup-hamming", type=int, default=10,
                    help="dHash 汉明距离阈值，仅保留与上一帧距离大于该值的帧"
                         " (默认 10)")
    ap.add_argument("--dedup-region", choices=list(REGION_FILTERS), default="board",
                    help="dHash 计算区域: board=板书区域(默认, 画面15%~60%高度), "
                         "full=全帧, top=上半部, center=中部")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="最大保留帧数 (默认 120)")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] 视频文件不存在: {args.video}", file=sys.stderr)
        return 2

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dedup":
        recs = extract_dedup(args.video, args.out_dir, args.dedup_fps,
                             args.dedup_hamming, args.max_frames, args.dedup_region)
        print(f"[ok] 提取 {len(recs)} 个去重帧 "
              f"(fps={args.dedup_fps}, hamming>{args.dedup_hamming}, "
              f"region={args.dedup_region}, cap={args.max_frames}) -> {args.out_dir}",
              file=sys.stderr)
    else:
        recs = extract_interval(args.video, args.out_dir, args.interval, args.fps)
        print(f"[ok] 提取 {len(recs)} 帧 -> {args.out_dir}", file=sys.stderr)

    manifest = args.out_dir / "frames.json"
    manifest.write_text(json.dumps(
        {"video": str(args.video.resolve()),
         "mode": args.mode,
         "interval": args.interval, "fps": args.fps,
         "dedup_fps": args.dedup_fps, "dedup_hamming": args.dedup_hamming,
         "dedup_region": args.dedup_region,
         "max_frames": args.max_frames,
         "count": len(recs), "frames": recs},
        ensure_ascii=False, indent=2,
    ))
    print(f"[ok] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
