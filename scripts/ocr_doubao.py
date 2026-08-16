#!/usr/bin/env python3
"""ocr_doubao.py — 使用豆包视觉模型 (doubao-seed-2-1-pro) 识别视频帧中的文字。

通过火山方舟 Ark Responses API 调用多模态视觉模型，
将帧图片中的文字、表格、公式等完整转写。

Pipeline:
    视频 → extract_frames.py 帧提取 → TOS 上传 → 逐帧调用 Ark 视觉模型 → 时间戳文字标注

Usage:
    python3 ocr_doubao.py --video slides.mp4 --out-dir out --prompt-ocr
    python3 ocr_doubao.py --manifest frames/frames.json --out-dir out --prompt-ocr

输出:
    captions.srt   # SRT 格式
    captions.json  # [{start, end, text}]
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

from config import (
    ARK_API_KEY,
    ARK_BASE_URL,
    ARK_EP_ID,
    validate_config,
)
from tos_upload import upload_file

# 两种 prompt 模式
PROMPT_CAPTION = (
    "用简洁的中文描述这张视频截图中的画面内容（不超过40字）。"
    "重点关注画面中的关键对象、文字、操作动作和图表。"
    "只输出描述本身，不要任何前缀。"
)

# 针对黑板/板书教学场景的 OCR 转写 prompt
PROMPT_OCR = (
    "你正在识别一段教学视频中黑板/白板上的板书内容。\n"
    "请逐字转写画面中黑板上的所有文字、数字和公式，原样输出，不要做任何总结、"
    "解释或补充。\n"
    "要求：\n"
    "1. 只转写黑板/白板上的板书内容，忽略画面中的人物、背景、讲台等非板书元素。\n"
    "2. 数学公式用 LaTeX 格式（如 $\\lim_{x \\to 0} \\frac{\\sin x}{x}$）输出。\n"
    "3. 中文文字保持原文，英文/数字原样输出，注意上下标（用 ^{} 和 _{} 表示）。\n"
    "4. 若板书分区域（如左右两块），用换行或分隔线保持原有布局顺序。\n"
    "5. 只输出转写内容本身，不要复述本指令，不要添加任何前缀。"
)

# 最后一帧没有下一帧时间戳可对齐，end 时间用固定时长补足
LAST_FRAME_DURATION = 2.0

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # 设置较长的请求超时，避免服务端长时间无响应导致挂起
        _client = OpenAI(base_url=ARK_BASE_URL, api_key=ARK_API_KEY,
                         timeout=180.0)
    return _client


def _to_data_uri(jpg_path: Path) -> str:
    """本地 JPG → Base64 data URI（官方文档 82379/1362931 推荐方式）。"""
    b64 = base64.b64encode(jpg_path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_vision_model(image: str, prompt: str, max_output_tokens: int = 8192) -> str:
    """调用 Ark 视觉模型（Responses API 格式）。

    image 参数支持两种形式（见火山方舟官方文档 82379/1362931）:
      - Base64 data URI:  data:image/jpeg;base64,<b64>   （本地小图，推荐）
      - 公网 URL:         https://...                    （如 TOS 预签名 URL）
    <10MB 两种方式均可。

    doubao-seed-2-1-pro 默认开启深度思考（thinking），逐帧 OCR 场景会导致
    单帧耗时 10+ 分钟，因此显式禁用 thinking 以保障批处理效率。
    """
    client = _get_client()

    try:
        response = client.responses.create(
            model=ARK_EP_ID,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image},
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            max_output_tokens=max_output_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as e:
        raise RuntimeError(f"Ark Responses API 错误: {e}") from e

    try:
        content = response.output_text
    except AttributeError:
        content = ""
    if not content:
        # 兜底：从 message 类型条目提取
        for item in response.output:
            if getattr(item, "type", "") == "message":
                for c in item.content:
                    if c.type == "output_text":
                        content = c.text
                        break
                if content:
                    break

    content = content or ""

    # 清理可能的 thinking 标签泄漏
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1].strip()

    return content.strip()


def frames_from(video: Path, out_dir: Path, *,
                max_frames: int = 120,
                density_sampling_fps: float = 1.0, density_floor: float = 0.3,
                density_min_increment: float = 0.35,
                density_fingerprint_hamming: int = 10,
                density_min_interval: float = 12.0,
                density_erase_drop: float = 2.0,
                density_erase_recover: float = 0.7,
                density_bright_threshold: int = 200) -> Path:
    """委托 extract_frames.py 进行帧提取（density 文字密度增量采样）。"""
    here = Path(__file__).resolve().parent
    cmd = [sys.executable, str(here / "extract_frames.py"),
           "--video", str(video), "--out-dir", str(out_dir / "frames"),
           "--density-sampling-fps", str(density_sampling_fps),
           "--density-floor", str(density_floor),
           "--density-min-increment", str(density_min_increment),
           "--density-fingerprint-hamming", str(density_fingerprint_hamming),
           "--density-min-interval", str(density_min_interval),
           "--density-erase-drop", str(density_erase_drop),
           "--density-erase-recover", str(density_erase_recover),
           "--density-bright-threshold", str(density_bright_threshold),
           "--max-frames", str(max_frames)]
    subprocess.run(cmd, check=True)
    return out_dir / "frames" / "frames.json"


def to_srt(records: list[dict]) -> str:
    def fmt(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, r in enumerate(records, 1):
        text = r["text"].replace("\n", " ") if "\n" in r["text"] else r["text"]
        blocks.append(f"{i}\n{fmt(r['start'])} --> {fmt(r['end'])}\n{text}\n")
    return "\n".join(blocks)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="豆包视觉模型 → 视频帧文字识别/描述")
    ap.add_argument("--video", required=False, type=Path,
                    help="输入视频（未指定 --manifest 时必填）")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="直接使用已有帧 manifest (如预处理后的 frames.json)，"
                         "跳过内部取帧")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--density-sampling-fps", type=float, default=1.0)
    ap.add_argument("--density-floor", type=float, default=0.3,
                    help="空板密度下限(%%), 低于该值视为空板不保留 (默认 0.3)")
    ap.add_argument("--density-min-increment", type=float, default=0.35)
    ap.add_argument("--density-fingerprint-hamming", type=int, default=10)
    ap.add_argument("--density-min-interval", type=float, default=12.0)
    ap.add_argument("--density-erase-drop", type=float, default=2.0)
    ap.add_argument("--density-erase-recover", type=float, default=0.7)
    ap.add_argument("--density-bright-threshold", type=int, default=200)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--prompt-ocr", action="store_true",
                    help="使用 OCR 转写模式（逐字转写所有文字/表格），"
                         "默认为画面描述模式")
    ap.add_argument("--image-mode", choices=["base64", "tos"], default="base64",
                    help="图片传入方式: base64=内联编码(默认,快), "
                         "tos=先上传 TOS 取公网 URL")
    args = ap.parse_args()

    if not args.manifest and not args.video:
        print("[err] 需指定 --video 或 --manifest", file=sys.stderr)
        return 2

    try:
        validate_config(service="vision")
    except RuntimeError as e:
        print(f"[err] {e}", file=sys.stderr)
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 帧提取：优先使用外部 manifest，否则从视频取帧
    if args.manifest:
        manifest = args.manifest
        if not manifest.is_file():
            print(f"[err] manifest 不存在: {manifest}", file=sys.stderr)
            return 2
    else:
        assert args.video is not None
        if not args.video.is_file():
            print(f"[err] 视频不存在: {args.video}", file=sys.stderr)
            return 2
        manifest = frames_from(args.video, args.out_dir,
                               max_frames=args.max_frames,
                               density_sampling_fps=args.density_sampling_fps,
                               density_floor=args.density_floor,
                               density_min_increment=args.density_min_increment,
                               density_fingerprint_hamming=args.density_fingerprint_hamming,
                               density_min_interval=args.density_min_interval,
                               density_erase_drop=args.density_erase_drop,
                               density_erase_recover=args.density_erase_recover,
                               density_bright_threshold=args.density_bright_threshold)
    frames = json.loads(manifest.read_text())["frames"]

    prompt = PROMPT_OCR if args.prompt_ocr else PROMPT_CAPTION
    mode_label = "OCR" if args.prompt_ocr else "caption"
    # OCR 转写模式允许更长输出（整面板书公式）
    max_output_tokens = 8192 if args.prompt_ocr else 1024

    print(f"[vision] {len(frames)} 帧, 模式={mode_label}, "
          f"图片={args.image_mode}, model={ARK_EP_ID}", file=sys.stderr)

    captions = []
    for i, fr in enumerate(frames):
        t0 = time.time()
        jpg_path = Path(fr["file"])

        try:
            if args.image_mode == "tos":
                image = upload_file(jpg_path)
            else:
                image = _to_data_uri(jpg_path)
            text = call_vision_model(image, prompt, max_output_tokens)
        except RuntimeError as e:
            print(f"[err] 帧 {i+1} 失败: {e}", file=sys.stderr)
            text = ""

        elapsed = time.time() - t0
        next_t = frames[i + 1]["t"] if i + 1 < len(frames) else fr["t"] + LAST_FRAME_DURATION
        captions.append({"start": fr["t"], "end": next_t, "text": text})

        preview = text.replace("\n", " ")[:60]
        print(f"[vision] {i+1}/{len(frames)} @ {fr['t']:.1f}s "
              f"({elapsed:.1f}s): {preview}", file=sys.stderr)

        # 简单限流：两次请求间间隔一小段
        if i < len(frames) - 1:
            time.sleep(0.3)

    (args.out_dir / "captions.srt").write_text(to_srt(captions), encoding="utf-8")
    (args.out_dir / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {len(captions)} 帧 → {args.out_dir}/captions.{{srt,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
