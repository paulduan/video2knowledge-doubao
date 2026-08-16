#!/usr/bin/env python3
"""merge_visual.py — 按时间戳融合 ASR 字幕和 OCR 视觉文字。

将 ASR 转写（语音内容）与 OCR 识别（画面文字）按时间对齐合并，
输出 merged.json 供 build_knowledge.py 使用。

对齐规则: 对每个 ASR 片段，找到其时间中点对应的 OCR 帧（±tolerance），
每个 OCR 帧最多绑定一个 ASR 片段（避免重复输出大段表格）。

Usage:
    python3 merge_visual.py --subtitles out/subtitles.json --visual out/captions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_segments(path: Path) -> list[dict]:
    """加载 {start, end, text} 列表。兼容 subtitles.json 和 captions.json。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("segments") or data.get("captions") or []


def find_visual_for(mid: float, visual: list[dict], used: set[int],
                    tolerance: float) -> dict | None:
    """返回时间窗覆盖 mid 的第一个未使用 OCR 帧。"""
    for i, vc in enumerate(visual):
        if i in used:
            continue
        if vc["start"] - tolerance <= mid <= vc["end"]:
            return vc
    return None


def merge(asr: list[dict], visual: list[dict], tolerance: float = 2.0) -> dict:
    segs = []
    used: set[int] = set()
    for a in asr:
        mid = (a["start"] + a["end"]) / 2
        vc = find_visual_for(mid, visual, used, tolerance)
        vtext = ""
        if vc is not None:
            vtext = vc["text"]
            used.add(visual.index(vc))
        segs.append({
            "start": a["start"], "end": a["end"],
            "text": a["text"], "visual": vtext,
        })
    return {
        "asr_count": len(asr),
        "visual_count": len(visual),
        "used_visual": len(used),
        "segments": segs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="融合 ASR 字幕与 OCR 视觉文字")
    ap.add_argument("--subtitles", required=True, type=Path)
    ap.add_argument("--visual", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="输出路径 (默认: <visual目录>/merged.json)")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="时间匹配容差秒数 (默认 2.0)")
    args = ap.parse_args()

    for p, name in [(args.subtitles, "subtitles"), (args.visual, "visual")]:
        if not p.is_file():
            print(f"[err] {name} 文件不存在: {p}", file=sys.stderr)
            return 2

    asr = load_segments(args.subtitles)
    visual = load_segments(args.visual)
    if not asr:
        print(f"[err] 无 ASR 片段: {args.subtitles}", file=sys.stderr)
        return 2
    print(f"[merge] {len(asr)} 个 ASR 片段 × {len(visual)} 个 OCR 帧",
          file=sys.stderr)

    result = merge(asr, visual, args.tolerance)
    result["asr_source"] = str(args.subtitles)
    result["visual_source"] = str(args.visual)
    result["visual_blocks"] = [
        {"start": v["start"], "end": v["end"], "text": v["text"]}
        for v in visual
    ]

    out = args.out or args.visual.parent / "merged.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 融合 {result['used_visual']}/{len(visual)} 个 OCR 帧 -> {out}",
          file=sys.stderr)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
