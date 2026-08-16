#!/usr/bin/env python3
"""batch_process.py — 批量处理 B 站缓存：全量转 mp4 → 逐视频 ASR + 取帧 + 预处理。

流程:
    1. 扫描缓存根目录，找出所有完整的 B 站缓存（视频+音频 m4s 齐全）
    2. 逐个缓存转换 mp4 → <输出根>/converted/<标题>.mp4
    3. 对每个 mp4 依次执行（单步失败不中断整体）:
        - 帧提取   extract_frames.py (默认 density 文字密度增量采样)
                  → <输出根>/<标题>/frames/
        - 帧预处理 preprocess_frames.py (黑板裁剪+增强) → <输出根>/<标题>/preprocessed/
        - ASR 语音 asr_doubao.py (默认关闭) → <输出根>/<标题>/subtitles.{srt,vtt,json}

输出目录固定为项目根目录下的 output/（--out-dir 可覆盖）。

默认行为（保守、省钱）:
    只执行本地免费步骤: 转换 mp4 + 取帧 + 预处理。
    ASR 是付费接口（按音频时长计费），批量跑全部 132 个视频会花很多时间和
    费用，因此需显式加 --with-asr 才开启。

可中断 / 可续跑:
    已存在的产物会自动跳过（复用 converted/ 下已转换的 mp4，以及每视频的
    frames.json / preprocessed/frames.json / subtitles.json），
    中断后重新执行同一命令即可续跑；加 --force 强制全部重跑。

Usage:
    # 预览将处理的缓存（不执行）
    python3 batch_process.py --list-only

    # 批量处理前 2 个缓存（仅本地步骤，不跑付费 ASR）
    python3 batch_process.py --limit 2

    # 完整流程（转换 + 取帧 + 预处理 + ASR），指定 cid
    python3 batch_process.py --with-asr --cid 25685856471

    # 全部 132 个缓存，完整流程（付费，谨慎）
    python3 batch_process.py --with-asr

    # 跳过预处理，只转换 + 取帧
    python3 batch_process.py --skip-preprocess
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convert_bilibili_cache import _safe_filename, convert, scan_caches
from run_pipeline import run_script, safe_dirname

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path("/Users/duanp/Movies/bilibili")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="批量处理 B 站缓存: 全量转 mp4 → 逐视频 取帧 + 预处理 + (可选) ASR")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help=f"B 站缓存根目录 (默认 {DEFAULT_CACHE_DIR})")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="输出根目录 (默认项目根目录下的 output/)")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个缓存，用于小规模调试 (默认处理全部)")
    ap.add_argument("--cid", action="append", default=[],
                    help="只处理指定的 cid，可多次指定; 默认处理全部缓存")
    ap.add_argument("--list-only", action="store_true",
                    help="只列出将处理的缓存，不执行任何转换/分析")
    ap.add_argument("--with-asr", action="store_true",
                    help="开启 ASR 语音转写（付费 API，按音频时长计费；默认关闭）")
    ap.add_argument("--language", default="zh-CN",
                    help="ASR 语言代码 (默认 zh-CN)")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="跳过帧预处理（黑板/白板裁剪 + 清晰化）")
    ap.add_argument("--no-crop", action="store_true",
                    help="预处理时不裁剪板面，只做增强")
    ap.add_argument("--force", action="store_true",
                    help="忽略已有产物，全部重跑")
    ap.add_argument("--density-sampling-fps", type=float, default=1.0,
                    help="density 模式密集采样率 fps (默认 1.0)")
    ap.add_argument("--density-floor", type=float, default=0.3,
                    help="density 模式空板密度下限(%%): 低于该值视为空板不保留"
                         " (默认 0.3)")
    ap.add_argument("--density-min-increment", type=float, default=0.35,
                    help="density 模式保留触发: 密度净增该值个百分点即保留 (默认 0.35)")
    ap.add_argument("--density-fingerprint-hamming", type=int, default=10,
                    help="density 模式指纹触发: 汉明距离 >= 该值即保留 (默认 10)")
    ap.add_argument("--density-min-interval", type=float, default=12.0,
                    help="density 模式保留帧最小间隔秒数 (默认 12.0)")
    ap.add_argument("--density-erase-drop", type=float, default=2.0,
                    help="density 模式擦板检测: 密度骤降该值个百分点视为擦板 (默认 2.0)")
    ap.add_argument("--density-erase-recover", type=float, default=0.7,
                    help="density 模式擦板恢复系数: 回升至擦前密度该比例即强制保留"
                         " (默认 0.7)")
    ap.add_argument("--density-bright-threshold", type=int, default=200,
                    help="density 模式亮像素阈值(0-255) (默认 200)")
    ap.add_argument("--density-min-frames", type=int, default=15,
                    help="density 模式低对比度兜底: 提取帧数低于该值且视频较长时, "
                         "自动降低亮像素阈值重跑 (默认 15)")
    ap.add_argument("--density-max-gap", type=float, default=300.0,
                    help="density 模式静止有板期兜底: 相邻保留帧间隔超过该秒数时, "
                         "补入后续首个有内容(密度>=floor)的采样帧 (默认 300, 0=关闭)")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="density 模式最大保留帧数 (默认 120)")
    return ap


def _frame_args(video: Path, out_dir: Path,
                args: argparse.Namespace) -> list[str]:
    """构造 extract_frames.py 的 density 参数（与 run_pipeline 保持一致）。"""
    a = ["--video", str(video), "--out-dir", str(out_dir)]
    a += ["--density-sampling-fps", str(args.density_sampling_fps),
          "--density-floor", str(args.density_floor),
          "--density-min-increment", str(args.density_min_increment),
          "--density-fingerprint-hamming", str(args.density_fingerprint_hamming),
          "--density-min-interval", str(args.density_min_interval),
          "--density-erase-drop", str(args.density_erase_drop),
          "--density-erase-recover", str(args.density_erase_recover),
          "--density-bright-threshold", str(args.density_bright_threshold),
          "--density-min-frames", str(args.density_min_frames),
          "--density-max-gap", str(args.density_max_gap),
          "--max-frames", str(args.max_frames)]
    return a


def _ensure_mp4(cache: dict, cache_dir: Path, converted_dir: Path,
                force: bool) -> Path:
    """确保 mp4 存在：已有则复用，否则调用 convert_bilibili_cache 转换。"""
    expected = converted_dir / f"{_safe_filename(cache['title'])}.mp4"
    if expected.is_file() and not force:
        print(f"[batch] 复用已转换 mp4: {expected}", file=sys.stderr)
        return expected
    print(f"[batch] 转换缓存 {cache['cid']} -> mp4 ...", file=sys.stderr)
    return convert(cache_dir / cache["cid"], converted_dir, force=force)


def main() -> int:
    args = build_parser().parse_args()
    base_dir = args.out_dir if args.out_dir is not None else HERE.parent / "output"
    converted_dir = base_dir / "converted"
    cache_dir = args.cache_dir

    # 0. 筛选待处理缓存
    try:
        caches = scan_caches(cache_dir)
    except RuntimeError as e:
        print(f"[err] {e}", file=sys.stderr)
        return 1
    caches = [c for c in caches if c["status"] == "ok"]
    if args.cid:
        available = {c["cid"] for c in caches}
        missing = [cid for cid in args.cid if cid not in available]
        if missing:
            print(f"[err] 未找到完整缓存: {', '.join(missing)}", file=sys.stderr)
            return 1
        caches = [c for c in caches if c["cid"] in set(args.cid)]
    if args.limit and args.limit > 0:
        caches = caches[:args.limit]

    if not caches:
        print("[err] 没有可处理的完整缓存", file=sys.stderr)
        return 1

    steps_txt = "转换 + 取帧 + 预处理" + (" + ASR" if args.with_asr else "")
    print(f"[batch] 待处理 {len(caches)} 个缓存: {steps_txt}", file=sys.stderr)
    print(f"[batch] 输出根目录: {base_dir}", file=sys.stderr)
    for i, c in enumerate(caches, 1):
        print(f"  {i:>3}. [{c['cid']}] {c['title']}")

    if args.list_only:
        return 0

    if args.with_asr:
        print("\n[注意] --with-asr 已开启，将调用付费 ASR 接口（按音频时长计费）。\n",
              file=sys.stderr)

    converted_dir.mkdir(parents=True, exist_ok=True)
    ok_list, fail_list = [], []
    total = len(caches)

    try:
        for i, c in enumerate(caches, 1):
            title = c["title"]
            print(f"\n{'='*60}\n[batch] {i}/{total} 正在处理: {title}\n{'='*60}",
                  file=sys.stderr)
            errors, done = [], []

            # 1. 转换 mp4（失败则跳过该视频）
            try:
                mp4 = _ensure_mp4(c, cache_dir, converted_dir, args.force)
                done.append("convert")
            except Exception as e:
                print(f"[err] {c['cid']} 转换失败: {e}", file=sys.stderr)
                errors.append(f"convert: {e}")
                fail_list.append((c["cid"], title, errors))
                continue

            out_dir = base_dir / safe_dirname(Path(mp4).stem)
            out_dir.mkdir(parents=True, exist_ok=True)

            # 2. 帧提取（density 文字密度增量采样）
            frames_manifest = out_dir / "frames" / "frames.json"
            if args.force or not frames_manifest.is_file():
                rc = run_script("extract_frames.py",
                                _frame_args(mp4, out_dir / "frames", args))
                if rc == 0:
                    done.append("frames")
                else:
                    errors.append(f"frames: rc={rc}")
            else:
                print(f"[batch] 复用帧提取: {frames_manifest}", file=sys.stderr)
                done.append("frames(复用)")

            # 3. 帧预处理（黑板/白板裁剪 + 清晰化）
            if not args.skip_preprocess:
                pre_manifest = out_dir / "preprocessed" / "frames.json"
                if args.force or not pre_manifest.is_file():
                    if not frames_manifest.is_file():
                        errors.append("preprocess: 缺少帧 manifest")
                    else:
                        rc = run_script("preprocess_frames.py", [
                            "--manifest", str(frames_manifest),
                            "--out-dir", str(out_dir / "preprocessed"),
                        ] + (["--no-crop"] if args.no_crop else []))
                        if rc == 0:
                            done.append("preprocess")
                        else:
                            errors.append(f"preprocess: rc={rc}")
                else:
                    print(f"[batch] 复用预处理: {pre_manifest}", file=sys.stderr)
                    done.append("preprocess(复用)")

            # 4. ASR 语音转写（付费 API，默认关闭）
            if args.with_asr:
                subs = out_dir / "subtitles.json"
                if args.force or not subs.is_file():
                    rc = run_script("asr_doubao.py", [
                        "--video", str(mp4), "--out-dir", str(out_dir),
                        "--language", args.language,
                    ])
                    if rc == 0:
                        done.append("asr")
                    else:
                        errors.append(f"asr: rc={rc}")
                else:
                    print(f"[batch] 复用 ASR 字幕: {subs}", file=sys.stderr)
                    done.append("asr(复用)")

            if errors:
                fail_list.append((c["cid"], title, errors))
            else:
                ok_list.append((c["cid"], title, done))

            print(f"\n[batch] {i}/{total} 完成: {title} -> {out_dir}"
                  f"  [{' + '.join(done)}]", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n[batch] 已被用户中断；已完成的部分下次运行会自动续跑。",
              file=sys.stderr)
        return 130

    # 汇总：成功/失败列表
    print(f"\n{'='*60}\n[batch] 汇总: 成功 {len(ok_list)} / 失败 {len(fail_list)}"
          f" / 共 {total}\n{'='*60}")
    for cid, title, done in ok_list:
        print(f"  [ok]   {cid}  {title}  ({' + '.join(done)})")
    for cid, title, errors in fail_list:
        print(f"  [err]  {cid}  {title}: {'; '.join(errors)}", file=sys.stderr)

    # 写一份汇总文件便于追溯
    summary = base_dir / "batch_summary.txt"
    lines = [f"批量处理汇总  (成功 {len(ok_list)} / 失败 {len(fail_list)} / 共 {total})",
             f"输出根目录: {base_dir}", ""]
    for cid, title, done in ok_list:
        lines.append(f"[ok]  {cid}  {title}  ({' + '.join(done)})")
    for cid, title, errors in fail_list:
        lines.append(f"[err] {cid}  {title}: {'; '.join(errors)}")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[batch] 汇总已写入: {summary}", file=sys.stderr)

    return 1 if fail_list else 0


if __name__ == "__main__":
    raise SystemExit(main())
