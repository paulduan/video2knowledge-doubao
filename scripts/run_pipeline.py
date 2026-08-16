#!/usr/bin/env python3
"""run_pipeline.py — 一键运行完整流水线。

支持两种输入:
  - 本地视频文件 (--video)
  - B 站客户端缓存 (--cid, 从缓存目录合并封装为 mp4 后处理)

流程:
    输入(本地或缓存) ─┬─→ 提取音频 → ASR 转写 ─┐ (并行)
                     └─→ 帧提取 → 预处理 → OCR ─┘
                          → 融合 → 豆包大模型总结 → 知识文档

path 含义:
    1 = 纯画面 OCR, 2 = 语音转写, 3 = 融合模式 (默认)

Usage:
    # 本地视频（默认输出到项目根目录的 output/<视频名>/）
    python3 run_pipeline.py --video lecture.mp4 --path 3

    # 指定输出基础目录
    python3 run_pipeline.py --video lecture.mp4 --out-dir output --path 3

    # B 站客户端缓存的视频（先列出缓存: convert_bilibili_cache.py --list）
    python3 run_pipeline.py --cid 25685856471 --path 3 --preprocess

    # 融合模式 + 完整转写（豆包大模型融合 语音+板书）
    python3 run_pipeline.py --video lecture.mp4 --out-dir output \
        --path 3 --preprocess --full-transcript

    # 默认 OCR 会逐字转写黑板板书；加 --caption 改为画面描述
    # 加 --full-transcript 在融合后调用豆包大模型生成完整转写（语音+板书融合）
"""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_script(script: str, args: list[str]) -> int:
    """运行同目录下的脚本。"""
    cmd = [sys.executable, str(HERE / script)] + args
    print(f"\n{'='*60}")
    print(f"[pipeline] 运行: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[err] {script} 返回错误码 {result.returncode}", file=sys.stderr)
    return result.returncode


def run_parallel(jobs: list[tuple[str, list[str]]]) -> bool:
    """并行执行多个脚本，全部成功返回 True。"""
    if len(jobs) < 2:
        return run_script(*jobs[0]) == 0

    print(f"\n[pipeline] 并行执行 {len(jobs)} 个任务...", file=sys.stderr)

    def _run(job):
        script, args = job
        return script, run_script(script, args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = [ex.submit(_run, j) for j in jobs]
        ok = True
        for f in concurrent.futures.as_completed(futures):
            script, rc = f.result()
            if rc != 0:
                ok = False
                print(f"[err] {script} 失败 (rc={rc})", file=sys.stderr)
    return ok


def convert_cache(cid: str, cache_dir: Path, out_dir: Path) -> str:
    """把 B 站缓存目录合并封装为 mp4，返回本地文件路径。

    以 stdout 最后一个路径为准。
    """
    log = out_dir / "convert.log"
    with log.open("w", encoding="utf-8") as f:
        cmd = [sys.executable, str(HERE / "convert_bilibili_cache.py"),
               "--cid", cid, "--cache-dir", str(cache_dir),
               "--out-dir", str(out_dir)]
        print(f"\n{'='*60}")
        print(f"[pipeline] 运行: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    lines = [l.strip() for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    if result.returncode != 0:
        print(f"[err] 缓存转换失败，详见 {log}", file=sys.stderr)
        raise RuntimeError(f"缓存转换失败 (cid={cid})")
    # stdout 中最后一个非日志行即 mp4 绝对路径
    for line in reversed(lines):
        p = Path(line)
        if p.is_file():
            return str(p)
    raise RuntimeError(f"缓存转换未返回视频路径，详见 {log}")


def safe_dirname(name: str) -> str:
    """把名称转成文件系统安全的目录名：替换非法字符并去掉首尾空白。"""
    cleaned = re.sub(r'[/\\:*?"<>|]', "_", name).strip()
    if not cleaned or cleaned in (".", ".."):
        return "output"
    return cleaned


def main() -> int:
    ap = argparse.ArgumentParser(description="video2knowledge 一键流水线")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="本地视频文件")
    src.add_argument("--cid", help="B 站客户端缓存目录名 (cid)，自动合并为 mp4")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="B 站缓存根目录 (默认 /Users/duanp/Movies/bilibili)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="输出基础目录；未指定时默认使用项目根目录下的 output/ 目录")
    ap.add_argument("--path", choices=["1", "2", "3"], default="3",
                    help="处理路径: 1=纯画面OCR, 2=语音转写, 3=融合模式 (默认 3)")
    ap.add_argument("--language", default="zh-CN",
                    help="ASR 语言代码 (默认 zh-CN)")
    ap.add_argument("--frame-mode", choices=["interval", "dedup"], default="dedup",
                    help="帧采样模式: dedup=密集采样+dHash去重(默认, 推荐), "
                         "interval=均匀间隔采样(可选)")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="interval 模式帧间隔秒数 (默认 2.0)")
    ap.add_argument("--dedup-fps", type=float, default=1.0,
                    help="dedup 模式密集采样率 fps (默认 1.0)")
    ap.add_argument("--dedup-hamming", type=int, default=10,
                    help="dedup 模式 dHash 汉明距离阈值 (默认 10)")
    ap.add_argument("--dedup-region", choices=["full", "top", "center", "board"],
                    default="board",
                    help="dedup 模式 dHash 计算区域: board=板书区域(默认), "
                         "full=全帧, top=上半部, center=中部")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="dedup 模式最大保留帧数 (默认 120)")
    ap.add_argument("--preprocess", action="store_true",
                    help="OCR 前预处理帧：提取黑板/白板区域 + 清晰化")
    ap.add_argument("--no-crop", action="store_true",
                    help="预处理时不裁剪板面，只做增强")
    ap.add_argument("--caption", action="store_true",
                    help="OCR 使用画面描述模式而非板书转写模式（默认逐字转写板书）")
    ap.add_argument("--full-transcript", action="store_true",
                    help="融合后额外调用豆包大模型生成完整转写 "
                         "(full_transcript.{json,srt,md})")
    args = ap.parse_args()

    # 输出基础目录：指定了 --out-dir 用该目录，否则用项目根目录下的 output/
    base_dir = args.out_dir if args.out_dir is not None else HERE.parent / "output"
    cache_dir = args.cache_dir if args.cache_dir is not None else Path(
        "/Users/duanp/Movies/bilibili")

    # 0. 输入源：B 站缓存 or 本地视频
    if args.cid:
        print(f"[pipeline] 转换 B 站缓存: {args.cid} "
              f"(来自 {cache_dir})", file=sys.stderr)
        try:
            video = convert_cache(args.cid, cache_dir, base_dir)
        except RuntimeError as e:
            print(f"[err] {e}", file=sys.stderr)
            return 2
        out_dir = base_dir / safe_dirname(Path(video).stem)
    else:
        if not args.video.is_file():
            print(f"[err] 视频文件不存在: {args.video}", file=sys.stderr)
            return 2
        video = str(args.video)
        out_dir = base_dir / safe_dirname(Path(video).stem)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] 输入视频: {video}", file=sys.stderr)
    print(f"[pipeline] 输出目录: {out_dir}", file=sys.stderr)

    # 1. 帧提取 + 预处理（OCR 前置，Path 1/3）
    frames_manifest: Path | None = None
    if args.path in ("1", "3"):
        frame_args = [
            "--video", video,
            "--out-dir", str(out_dir / "frames"),
            "--mode", args.frame_mode,
        ]
        if args.frame_mode == "interval":
            frame_args += ["--interval", str(args.interval)]
        else:
            frame_args += [
                "--dedup-fps", str(args.dedup_fps),
                "--dedup-hamming", str(args.dedup_hamming),
                "--dedup-region", args.dedup_region,
                "--max-frames", str(args.max_frames),
            ]
        rc = run_script("extract_frames.py", frame_args)
        if rc != 0:
            return rc
        frames_manifest = out_dir / "frames" / "frames.json"

        if args.preprocess:
            if not frames_manifest.is_file():
                print(f"[err] 帧 manifest 不存在: {frames_manifest}", file=sys.stderr)
                return 2
            pre_dir = out_dir / "preprocessed"
            rc = run_script("preprocess_frames.py", [
                "--manifest", str(frames_manifest),
                "--out-dir", str(pre_dir),
            ] + (["--no-crop"] if args.no_crop else []))
            if rc != 0:
                return rc
            frames_manifest = pre_dir / "frames.json"

    # 2. 并行执行 ASR + OCR
    jobs: list[tuple[str, list[str]]] = []
    if args.path in ("2", "3"):
        jobs.append(("asr_doubao.py", [
            "--video", video, "--out-dir", str(out_dir),
            "--language", args.language,
        ]))
    if args.path in ("1", "3"):
        ocr_args = [
            "--out-dir", str(out_dir),
            "--mode", args.frame_mode,
        ]
        if args.frame_mode == "interval":
            ocr_args += ["--interval", str(args.interval)]
        else:
            ocr_args += [
                "--dedup-fps", str(args.dedup_fps),
                "--dedup-hamming", str(args.dedup_hamming),
                "--dedup-region", args.dedup_region,
                "--max-frames", str(args.max_frames),
            ]
        if not args.caption:
            ocr_args += ["--prompt-ocr"]  # 默认逐字转写黑板板书
        if frames_manifest is not None:
            ocr_args += ["--manifest", str(frames_manifest)]
        else:
            ocr_args += ["--video", video]
        jobs.append(("ocr_doubao.py", ocr_args))

    if not jobs:
        print("[err] 无任务可执行", file=sys.stderr)
        return 2

    if not run_parallel(jobs):
        print("[err] 并行任务存在失败", file=sys.stderr)
        return 1

    # 3. 融合 (Path 3)
    if args.path == "3":
        rc = run_script("merge_visual.py", [
            "--subtitles", str(out_dir / "subtitles.json"),
            "--visual", str(out_dir / "captions.json"),
        ])
        if rc != 0:
            return rc

    # 3.5 完整转写（可选，豆包大模型融合语音+板书）
    if args.full_transcript and args.path in ("2", "3"):
        ft_args = ["--out-dir", str(out_dir)]
        if args.path == "3" and (out_dir / "merged.json").is_file():
            ft_args += ["--merged", str(out_dir / "merged.json")]
        else:
            ft_args += ["--subtitles", str(out_dir / "subtitles.json")]
        rc = run_script("full_transcript.py", ft_args)
        if rc != 0:
            return rc

    # 4. 构建知识文档（豆包大模型总结）
    if args.path == "1":
        subtitles_path = str(out_dir / "captions.json")
    else:
        subtitles_path = str(out_dir / "subtitles.json")

    knowledge_args = [
        "--subtitles", subtitles_path,
        "--out-dir", str(out_dir),
    ]
    if args.path == "3":
        knowledge_args += ["--merged", str(out_dir / "merged.json")]

    rc = run_script("build_knowledge.py", knowledge_args)
    if rc != 0:
        return rc

    print(f"\n{'='*60}")
    print(f"[pipeline] 完成！输出目录: {out_dir}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
