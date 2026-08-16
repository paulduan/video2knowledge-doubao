#!/usr/bin/env python3
"""extract_frames.py — 从视频中采样代表性帧。

三种模式:
  - interval (可选): 均匀 fps 采样
  - dedup (可选): 密集采样 + 感知哈希(dHash)去重，适合板书/录屏类视频
  - density (默认): 文字密度增量采样，以板区文字密度为主信号、内容指纹为辅

density 模式设计依据（对真实教学视频的实测，见 analyze_frame_dedup.py 与
density 数据标定）:
  - 板区文字密度 = 板书/粉笔亮像素占板区比例。实测范围:
    空板 ≈ 0.06%~0.13%，有板书 0.3%~4%。密度随书写单调递增，
    是"板书是否在变化"的可靠信号。
  - 单纯 dHash 去重在三个场景下失真:
      1. 跳跃大: 教师长时间只讲不写 → 板面不变 → 不取帧，内容突然出现时
         出现数百秒空档（实测最大 438s），渐进书写过程被整体跳过。
      2. 开头空板多帧: 空板时教师走动/路人经过也会触发 dHash 变化，
         开头 70s 被捕获 9 个几乎无内容的帧。
      3. 过密: 连续书写时每笔都触发 dHash 阈值，帧间隔仅 1~3s。
  - density 模式对策:
      1. 密度相对上一保留帧净增 min_increment 个百分点 → 保留（捕捉书写增量）
      2. 密度相近但指纹汉明距离 >= fingerprint_hamming → 保留
         （改写/换公式等不显著改变密度的内容变化）
      3. 密度低于 floor 视为空板，一律不保留（修掉问题 2）
      4. 保留帧最小间隔 min_interval 秒（修掉问题 3）
      5. 擦板重写: 检测"密度骤降(擦板) → 回升"，回升至擦前密度
         erase_recover_frac 即强制保留一帧（修掉问题 4）
      6. 低对比度兜底: 提取帧数 < min_frames 且视频较长时自动降低亮像素阈值
         重跑（粉笔亮度偏低的视频默认阈值会把密度信号压平，几乎取不到帧）
      7. 静止有板期兜底: 相邻保留帧间隔 > max_gap 秒时补入后续首个有内容的
         采样帧（教师长时间只讲不写、板面静止会产生数百秒大空档）
  - 指纹沿用 64-bit dHash；density 模式单条 ffmpeg 管道流式输出板区
    原始分辨率灰度图，同时计算密度与指纹（Python 内降采样到 9x8），
    避免全分辨率整批载入内存，也不额外增加 ffmpeg 解码遍数。

输出: <out_dir>/frame_%06d.jpg + frames.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
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

# density 模式板区灰度滤镜：与 REGION_FILTERS["board"] 的 crop 完全一致，
# 但保留原始分辨率 —— 密度统计需要真实亮度分布，缩放到 9x8 会毁掉它。
BOARD_CROP_GRAY = "crop=iw:ih*0.45:0:ih*0.15,format=gray"


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


def _analyze_all_frames(video: Path, fps: float, crop_filter: str,
                        bright_threshold: int) -> list[tuple[int, float, np.ndarray]]:
    """对视频以 fps 采样，返回 [(帧索引, 板区密度%, dHash)] 列表。

    单条 ffmpeg 管道: fps 采样 + 裁剪到板区(保留原始分辨率) + 灰度 PGM，
    流式逐帧解析（不整体载入内存，避免全分辨率数据占满内存）。
    每帧计算:
      - density: 板区像素中亮度 > bright_threshold 的占比(%)，反映板书量
      - dHash:   Python 内降采样到 9x8 后计算，与 dedup 模式同一定义
    """
    vf = f"fps={fps},{crop_filter}"
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", vf, "-f", "image2pipe", "-vcodec", "pgm", "-"],
        stdout=subprocess.PIPE,
    )
    out: list[tuple[int, float, np.ndarray]] = []
    idx = 0
    aborted = False  # 帧数据截断/损坏：需要终止 ffmpeg，避免管道写满挂死
    try:
        buf = proc.stdout
        while True:
            magic = buf.readline()
            if not magic or magic.strip() != b"P5":
                break
            dims = buf.readline()
            if not dims:
                break
            try:
                w, h = (int(x) for x in dims.split())
            except ValueError:
                break
            buf.readline()  # maxval 行（通常 "255"）
            data = buf.read(w * h)
            if len(data) != w * h:
                aborted = True
                break
            pix = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            density = float((pix > bright_threshold).mean() * 100.0)
            small = cv2.resize(pix, (9, 8), interpolation=cv2.INTER_AREA)
            out.append((idx, density, _dhash_bits(small)))
            idx += 1
    finally:
        if aborted:
            proc.kill()
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
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


def _select_density_frames(samples: list[tuple[int, float, np.ndarray]], *,
                           fps: float, floor: float, min_increment: float,
                           fingerprint_hamming: int, min_interval: float,
                           erase_drop: float, erase_recover_frac: float,
                           max_frames: int, max_gap: float
                           ) -> list[tuple[int, float, np.ndarray, int | None, bool]]:
    """从密度采样序列中挑选保留帧（含 max-gap 兜底与帧数上限）。

    返回 [(idx, 密度%, dHash, 相对上一保留帧hamming, 是否强制帧)]，按时间升序。
    选择规则:
      1. 密度相对上一保留帧净增 >= min_increment 个百分点 → 保留（书写增量）
      2. 密度相近但指纹汉明距离 >= fingerprint_hamming → 保留（改写/换公式）
      3. 密度 < floor 视为空板，一律不保留（修掉"开头空板多帧"）
      4. 保留帧最小间隔 min_interval 秒（修掉连续书写的 1~3s 过密）
      5. 检测"密度骤降(擦板) → 回升": 回升至 erase_recover_frac * 擦前密度
         即强制保留一帧（防止整板重写内容被跳过）
    最后均匀降采样到 max_frames 上限。
    """
    def t_of(i: int) -> float:
        return round(i / fps, 3)

    kept: list[tuple[int, float, np.ndarray, int | None, bool]] = []
    baseline_density: float | None = None     # 上一保留帧的密度
    last_kept_idx: int | None = None
    last_kept_hash: np.ndarray | None = None
    erased_t: float | None = None             # 检测到擦板的时间（进入恢复期）
    erased_baseline: float | None = None      # 擦板前的密度

    for idx, d, h in samples:
        t = idx / fps
        # 空板一律不保留（含开头空板期）
        if d < floor:
            continue
        # 全局最小保留间隔
        if last_kept_idx is not None and t - t_of(last_kept_idx) < min_interval:
            continue

        keep = False
        if erased_t is not None:
            # 恢复期: 只在密度回升到擦前水平后强制保留一帧
            if d >= erase_recover_frac * erased_baseline:
                keep = True
                erased_t = None
                erased_baseline = None
        else:
            if baseline_density is None:
                keep = True  # 空板结束后的首帧（进入正文期）
            elif d - baseline_density >= min_increment:
                keep = True  # 密度净增 -> 书写增量
            elif (last_kept_hash is not None
                  and _hamming(h, last_kept_hash) >= fingerprint_hamming):
                keep = True  # 内容指纹变化大 -> 改写/换公式等
            # 擦板检测: 密度骤降（如整板擦掉）→ 进入恢复期，且本帧不保留
            if baseline_density is not None and d < baseline_density - erase_drop:
                erased_t, erased_baseline = t, baseline_density
                keep = False

        if keep:
            ham = None if last_kept_hash is None else _hamming(h, last_kept_hash)
            kept.append((idx, d, h, ham, False))
            baseline_density = d
            last_kept_idx = idx
            last_kept_hash = h

    # max-gap 兜底: 静止有板期（教师长时间只讲不写、板面满而静止）会产生数百秒
    # 大空档; 相邻保留帧间隔超过 max_gap 秒时，补入 interval 之后首个 density>=floor
    # 的采样帧（空板期 density<floor 仍跳过，保持"跳过死时间"行为）。补入后以新的
    # 最后保留帧为基准继续检查，可在一个大空档内连续补入多帧。
    if max_gap > 0 and len(kept) >= 2:
        final: list[tuple[int, float, np.ndarray, int | None, bool]] = [kept[0]]
        si, n = 0, len(samples)
        for nxt in kept[1:]:
            nxt_t = nxt[0] / fps
            while True:
                last = final[-1]
                last_t = last[0] / fps
                if nxt_t - last_t < max_gap:
                    break
                target = last_t + max_gap
                # samples 指针推进到第一个 t >= target 的采样帧
                while si < n and samples[si][0] / fps < target:
                    si += 1
                # 在 [target, nxt_t) 内找首个有内容的采样帧
                forced: tuple[int, float, np.ndarray] | None = None
                j = si
                while j < n and samples[j][0] / fps < nxt_t:
                    if samples[j][1] >= floor:
                        forced = samples[j]
                        break
                    j += 1
                if forced is None:
                    break  # 该区间为空板期，保持跳过
                si = j + 1
                fidx, fd, fh = forced
                fham = _hamming(fh, last[2])
                final.append((fidx, fd, fh, fham, True))
            final.append(nxt)
        kept = final

    if len(kept) > max_frames:
        step = len(kept) / max_frames
        kept = [kept[int(j * step)] for j in range(max_frames)]
    return kept


def extract_density(video: Path, out_dir: Path, *, fps: float, floor: float,
                    min_increment: float, fingerprint_hamming: int,
                    min_interval: float, erase_drop: float,
                    erase_recover_frac: float, bright_threshold: int,
                    max_frames: int, min_frames: int = 15,
                    max_gap: float = 300.0) -> tuple[list[dict], int]:
    """文字密度增量采样（density 模式）。

    以板区文字密度为主信号、内容指纹为辅决定保留哪些帧（选择规则见
    _select_density_frames）。最后均匀降采样到 max_frames 上限。
    帧记录附带 density（板区密度%）与 hamming（相对上一保留帧的指纹距离）。

    两个兜底机制（默认开启）:
      - 低对比度兜底: 粉笔亮度偏低的视频在默认阈值下密度信号被压平（如整个
        48 分钟只取 1 帧）; 若一次提取帧数 < min_frames 且视频时长 > 600s，
        依次用更低阈值 180→160→150 重跑，取首个帧数达标或帧数最多的结果，
        并把实际使用的阈值随 manifest 记录。降级时向 stderr 打印提示。
      - 静止有板期兜底: 相邻保留帧间隔超过 max_gap 秒时补入强制帧（见
        _select_density_frames 的 max-gap 逻辑），强制帧以 "forced": true 标记。
    返回 (帧记录列表, 实际使用的亮像素阈值)。
    """
    duration = ffprobe_duration(video)
    thresholds = [bright_threshold]
    if duration > 600.0:
        thresholds += [180, 160, 150]

    best: list[tuple[int, float, np.ndarray, int | None, bool]] | None = None
    best_threshold = bright_threshold
    chosen: list[tuple[int, float, np.ndarray, int | None, bool]] | None = None
    chosen_threshold = bright_threshold
    first_count = 0  # 原阈值下提取的帧数（用于降级日志）

    for th in thresholds:
        samples = _analyze_all_frames(video, fps, BOARD_CROP_GRAY, th)
        if not samples:
            continue
        kept = _select_density_frames(
            samples, fps=fps, floor=floor, min_increment=min_increment,
            fingerprint_hamming=fingerprint_hamming, min_interval=min_interval,
            erase_drop=erase_drop, erase_recover_frac=erase_recover_frac,
            max_frames=max_frames, max_gap=max_gap)
        if th == bright_threshold:
            first_count = len(kept)
        if best is None or len(kept) > len(best):
            best, best_threshold = kept, th
        if len(kept) >= min_frames:
            chosen, chosen_threshold = kept, th
            break

    if chosen is None:
        chosen, chosen_threshold = best, best_threshold
    if chosen is None:
        return [], bright_threshold

    if chosen_threshold != bright_threshold:
        print(f"[density] 亮度阈值降至 {chosen_threshold} (原 {bright_threshold})，"
              f"帧数 {first_count}→{len(chosen)}", file=sys.stderr)

    records = []
    for n, (fi, d, h, ham, forced) in enumerate(chosen, 1):
        jpg = out_dir / f"frame_{n:06d}.jpg"
        t = round(fi / fps, 3)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", str(t), "-i", str(video), "-frames:v", "1",
             "-q:v", "3", "-y", str(jpg)],
            check=True,
        )
        if jpg.exists():
            rec = {"file": str(jpg), "t": t, "density": round(d, 3)}
            if ham is not None:
                rec["hamming"] = ham
            if forced:
                rec["forced"] = True
            records.append(rec)
    return records, chosen_threshold


def main() -> int:
    ap = argparse.ArgumentParser(description="视频帧采样工具")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["interval", "dedup", "density"],
                    default="density",
                    help="帧采样模式: density=文字密度增量采样(默认, 推荐), "
                         "dedup=dHash去重, interval=均匀间隔")
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
                    help="dHash 计算区域: board=板书区域(默认, 画面15%%~60%%高度), "
                         "full=全帧, top=上半部, center=中部")
    ap.add_argument("--density-sampling-fps", type=float, default=1.0,
                    help="density 模式密集采样率 (默认 1.0 fps)")
    ap.add_argument("--density-floor", type=float, default=0.3,
                    help="density 模式空板密度下限(%%): 板区密度低于该值视为"
                         "空板, 一律不保留 (默认 0.3)")
    ap.add_argument("--density-min-increment", type=float, default=0.35,
                    help="density 模式保留触发: 密度相对上一保留帧净增该值个"
                         "百分点即保留 (默认 0.35)")
    ap.add_argument("--density-fingerprint-hamming", type=int, default=10,
                    help="density 模式指纹触发: 密度相近但指纹汉明距离 >= 该值"
                         "即保留 (默认 10)")
    ap.add_argument("--density-min-interval", type=float, default=12.0,
                    help="density 模式保留帧最小间隔秒数 (默认 12.0)")
    ap.add_argument("--density-erase-drop", type=float, default=2.0,
                    help="density 模式擦板检测: 密度相对上一保留帧骤降该值个"
                         "百分点视为擦板 (默认 2.0)")
    ap.add_argument("--density-erase-recover", type=float, default=0.7,
                    help="density 模式擦板恢复系数: 擦板后密度回升至擦前密度的"
                         "该比例即强制保留一帧 (默认 0.7)")
    ap.add_argument("--density-bright-threshold", type=int, default=200,
                    help="density 模式亮像素阈值(0-255): 板区灰度高于该值计为"
                         "板书/粉笔像素 (默认 200)")
    ap.add_argument("--density-min-frames", type=int, default=15,
                    help="density 模式低对比度兜底: 一次提取帧数低于该值且视频"
                         "时长>600s 时, 自动降低亮像素阈值重跑 (默认 15)")
    ap.add_argument("--density-max-gap", type=float, default=300.0,
                    help="density 模式静止有板期兜底: 相邻保留帧间隔超过该秒数时, "
                         "补入后续首个有内容(密度>=floor)的采样帧 (默认 300, 0=关闭)")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="dedup/density 模式最大保留帧数 (默认 120)")
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
        bright_used = args.density_bright_threshold
        print(f"[ok] 提取 {len(recs)} 个去重帧 "
              f"(fps={args.dedup_fps}, hamming>{args.dedup_hamming}, "
              f"region={args.dedup_region}, cap={args.max_frames}) -> {args.out_dir}",
              file=sys.stderr)
    elif args.mode == "density":
        recs, bright_used = extract_density(
            args.video, args.out_dir,
            fps=args.density_sampling_fps, floor=args.density_floor,
            min_increment=args.density_min_increment,
            fingerprint_hamming=args.density_fingerprint_hamming,
            min_interval=args.density_min_interval,
            erase_drop=args.density_erase_drop,
            erase_recover_frac=args.density_erase_recover,
            bright_threshold=args.density_bright_threshold,
            max_frames=args.max_frames,
            min_frames=args.density_min_frames,
            max_gap=args.density_max_gap,
        )
        print(f"[ok] 提取 {len(recs)} 个密度增量帧 "
              f"(fps={args.density_sampling_fps}, floor={args.density_floor}%%, "
              f"inc>={args.density_min_increment}%%, "
              f"hamming>={args.density_fingerprint_hamming}, "
              f"min_interval={args.density_min_interval}s, "
              f"erase_drop={args.density_erase_drop}%%, "
              f"bright={bright_used}, max_gap={args.density_max_gap}s, "
              f"cap={args.max_frames}) -> {args.out_dir}", file=sys.stderr)
    else:
        recs = extract_interval(args.video, args.out_dir, args.interval, args.fps)
        bright_used = args.density_bright_threshold
        print(f"[ok] 提取 {len(recs)} 帧 -> {args.out_dir}", file=sys.stderr)

    manifest = args.out_dir / "frames.json"
    manifest.write_text(json.dumps(
        {"video": str(args.video.resolve()),
         "mode": args.mode,
         "interval": args.interval, "fps": args.fps,
         "dedup_fps": args.dedup_fps, "dedup_hamming": args.dedup_hamming,
         "dedup_region": args.dedup_region,
         "density_sampling_fps": args.density_sampling_fps,
         "density_floor": args.density_floor,
         "density_min_increment": args.density_min_increment,
         "density_fingerprint_hamming": args.density_fingerprint_hamming,
         "density_min_interval": args.density_min_interval,
         "density_erase_drop": args.density_erase_drop,
         "density_erase_recover": args.density_erase_recover,
         "density_bright_threshold": args.density_bright_threshold,
         "density_bright_threshold_used": bright_used,
         "density_min_frames": args.density_min_frames,
         "density_max_gap": args.density_max_gap,
         "max_frames": args.max_frames,
         "count": len(recs), "frames": recs},
        ensure_ascii=False, indent=2,
    ))
    print(f"[ok] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
