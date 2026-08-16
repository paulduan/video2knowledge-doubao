#!/usr/bin/env python3
"""full_transcript.py — 用豆包大模型生成融合语音+板书的完整转写。

与 build_full_transcript.py（纯本地拼接 Markdown）不同，本脚本调用豆包大模型，
把 ASR 语音转写与 OCR 板书内容按时间窗分组喂给模型，由模型输出融合后的完整转写：
  - 板书内容补充语音中没有讲清楚的地方；
  - 两种信息互相印证，纠正 OCR/ASR 的错别字与误读；
  - 输出自然流畅的中文。

流程:
    输入(merged.json 或 subtitles+visual)
      → 按时间窗分块 (默认 90s/窗)
      → 每块构建 prompt（语音片段 + 该时间窗内板书）
      → 豆包模型逐段返回融合文本（JSON: [{i, text}, ...]）
      → 保留原始时间戳写出 full_transcript.{json,srt,md}

若未配置 ARK_API_KEY 或单窗调用失败，自动退化为机械融合
（板书文本拼接在语音文本之后），保证流水线不中断。

Usage:
    python3 full_transcript.py --merged out/merged.json --out-dir out
    python3 full_transcript.py --subtitles out/subtitles.json --visual out/captions.json --out-dir out
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from config import (
    ARK_API_KEY,
    ARK_URL,
    ARK_TEXT_EP_ID,
    ARK_EP_ID,
)

# 文本模型接入点：优先文本模型，没有则复用视觉模型接入点
_TEXT_MODEL = ARK_TEXT_EP_ID or ARK_EP_ID

# 单次 prompt 最大字符数（防止超 token 上限）
MAX_PROMPT_CHARS = 6000

SYSTEM_PROMPT = (
    "你是视频课程内容整理助手。你同时拥有语音转写（ASR）与黑板板书（OCR）"
    "两种信息源，需要把它们融合成一份通顺、准确的完整转写。"
)

TASK_PROMPT = """下面是一段时间窗内的一段教学视频内容，包含:
1. 【语音】每行形如 `i. [mm:ss] 语音文本`，i 是语音片段的编号
2. 【板书】该时间窗内出现的黑板/白板板书内容（可能有 0~多块）

请把语音和板书两种信息融合，为【每一条语音片段 i】生成一段润色后的完整转写文本：
1. 板书内容可以补充语音中没有讲清楚或没讲到的关键点；
2. 两种信息互相印证：利用板书纠正语音的错别字/谐音错误，利用语音纠正板书的误读；
3. 输出自然流畅的中文口语，同一片段内语句连贯，不要生硬拼接；
4. 每条文本长度与对应语音原文相当（可用板书补充，但不要大幅扩写）；
5. 编号必须对应回 i，缺失的编号跳过。

只输出一个 JSON 数组，不要输出任何解释或其他文字，格式如下:
[{"i": 0, "text": "..."}, {"i": 1, "text": "..."}]

输入内容:
"""


def fmt_mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def fmt_ts(sec: float, sep: str = ",") -> str:
    """秒 → SRT 时间戳 (HH:MM:SS,mmm)。"""
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def load_segments(path: Path) -> list[dict]:
    """加载 {start, end, text} 列表，兼容 subtitles/captions/merged.json。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("segments") or data.get("captions") or []


def merge_local(asr: list[dict], visual: list[dict],
                tolerance: float = 2.0) -> list[dict]:
    """按时间把 OCR 板书块绑定到 ASR 片段（与 merge_visual.py 逻辑一致）。"""
    segs = []
    used: set[int] = set()
    for a in asr:
        mid = (a["start"] + a["end"]) / 2
        vc = None
        for i, v in enumerate(visual):
            if i in used:
                continue
            if v["start"] - tolerance <= mid <= v["end"]:
                vc = v
                used.add(i)
                break
        segs.append({
            "start": a["start"], "end": a["end"],
            "text": a["text"], "visual": vc["text"] if vc else "",
        })
    return segs


def chunk_by_window(segs: list[dict], window: float) -> list[list[dict]]:
    """按时间窗把连续语音片段分组。每窗约 window 秒。"""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    w_end = None
    for s in segs:
        if w_end is None or s["start"] >= w_end:
            if cur:
                chunks.append(cur)
            w_end = s["start"] + window
            cur = [s]
        else:
            cur.append(s)
    if cur:
        chunks.append(cur)
    return chunks


def blocks_in_window(blocks: list[dict], start: float, end: float) -> list[dict]:
    """返回时间窗 [start, end] 内有交集的板书块。"""
    return [b for b in blocks if b["end"] >= start and b["start"] <= end]


def build_prompt(chunk: list[dict], blocks: list[dict]) -> str:
    """构建单窗 prompt：语音片段（带编号）+ 该窗内板书。"""
    lines = []
    for i, s in enumerate(chunk):
        ts = fmt_mmss(s["start"])
        text = (s.get("text") or "").strip()
        lines.append(f"{i}. [{ts}] {text}")
    if blocks:
        lines.append("")
        lines.append("【板书】")
        for b in blocks:
            ts = fmt_mmss(b["start"])
            text = (b.get("text") or "").strip()
            lines.append(f"- [{ts}] {text}")
    raw = "\n".join(lines)
    return TASK_PROMPT + raw[:MAX_PROMPT_CHARS]


def ask_doubao(prompt: str, system: str = SYSTEM_PROMPT,
               max_tokens: int = 4096, timeout: int = 60) -> str | None:
    """调用豆包大模型 API（Ark OpenAI 兼容 chat/completions 格式）。"""
    if not ARK_API_KEY:
        return None

    payload = {
        "model": _TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        # doubao-seed-2-1-pro 默认深度思考，会导致响应耗时极长。
        # 批处理场景显式禁用 thinking 以保障速度。
        "thinking": {"type": "disabled"},
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ARK_API_KEY}",
    }

    req = urllib.request.Request(ARK_URL, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return None
    except (urllib.error.URLError, OSError):
        return None


def parse_fusion_response(resp: str) -> list[dict] | None:
    """解析模型返回的 [{i, text}, ...] JSON，失败返回 None。"""
    cleaned = re.sub(r"^```[a-zA-Z]*\s*\n?", "", resp.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", cleaned, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("text"):
            out.append({"i": item.get("i"), "text": str(item["text"]).strip()})
    return out or None


def mechanical_fusion(seg: dict) -> str:
    """机械融合兜底：板书文本拼接在语音文本之后。"""
    t = (seg.get("text") or "").strip()
    v = (seg.get("visual") or "").strip()
    if t and v:
        return f"{t}（板书: {v}）"
    return t or v


def render_md(title: str, source: str, duration: float, sections: list[tuple],
              meta: str) -> str:
    """组装 Markdown：头部 + 每窗一节。"""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {title}",
        "",
        "> 完整转写（豆包大模型融合 语音 + 板书）",
        f"> 来源: {source} | 视频时长: {fmt_mmss(duration)} | 生成时间: {now}",
        f"> {meta}",
        "",
    ]
    for sec_start, sec_end, body in sections:
        lines.append(f"## {fmt_mmss(sec_start)} – {fmt_mmss(sec_end)}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def escape_md_dollars(text: str) -> str:
    """转义 Markdown 中未被转义的 $（OCR 板书 LaTeX 公式定界符），避免渲染异常。

    只转义前面没有反斜杠的独立 $；已转义的 \\$ 保持原样，防止二次转义。
    """
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="豆包大模型 → 融合语音+板书的完整转写")
    ap.add_argument("--merged", type=Path, default=None,
                    help="merge_visual.py 输出的 merged.json（推荐）")
    ap.add_argument("--subtitles", type=Path, default=None,
                    help="ASR 语音转写 subtitles.json")
    ap.add_argument("--visual", type=Path, default=None,
                    help="OCR 板书 captions.json（可选）")
    ap.add_argument("--out-dir", required=True, type=Path, help="输出目录")
    ap.add_argument("--window", type=float, default=90.0,
                    help="时间窗长度（秒），每窗一次模型调用 (默认 90)")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="调试用：最多处理前 N 个时间窗 (默认 0=全部)")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="语音-板书时间匹配容差秒数 (默认 2.0)")
    ap.add_argument("--title", default=None, help="文档标题（默认取目录名）")
    args = ap.parse_args()

    # ---- 加载输入 ----
    segs: list[dict] = []
    blocks: list[dict] = []
    source = ""
    if args.merged:
        if not args.merged.is_file():
            print(f"[err] merged.json 不存在: {args.merged}", file=sys.stderr)
            return 2
        data = json.loads(args.merged.read_text(encoding="utf-8"))
        segs = data.get("segments") or []
        blocks = data.get("visual_blocks") or []
        source = args.merged.name
        for s in segs:
            s.setdefault("visual", "")
    elif args.subtitles:
        if not args.subtitles.is_file():
            print(f"[err] 字幕文件不存在: {args.subtitles}", file=sys.stderr)
            return 2
        asr = load_segments(args.subtitles)
        if args.visual:
            if not args.visual.is_file():
                print(f"[err] 板书文件不存在: {args.visual}", file=sys.stderr)
                return 2
            visual = load_segments(args.visual)
            segs = merge_local(asr, visual, args.tolerance)
            blocks = [{"start": v["start"], "end": v["end"], "text": v["text"]}
                      for v in visual]
            source = f"{args.subtitles.name} + {args.visual.name}"
        else:
            segs = [{**s, "visual": ""} for s in asr]
            source = args.subtitles.name
    else:
        ap.error("需指定 --merged，或 --subtitles (+可选 --visual)")

    if not segs:
        print("[err] 无语音片段可处理", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 按时间窗分块 ----
    chunks = chunk_by_window(segs, args.window)
    total_chunks = len(chunks)
    if args.max_chunks and args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    print(f"[ft] {len(segs)} 段语音 × {len(blocks)} 块板书 → "
          f"{total_chunks} 个时间窗（{args.window:.0f}s/窗，本次处理 "
          f"{len(chunks)} 个）", file=sys.stderr)
    print(f"[ft] 模型: {_TEXT_MODEL} | API: {'已配置' if ARK_API_KEY else '未配置(机械融合)'}",
          file=sys.stderr)

    # ---- 逐窗调用豆包融合 ----
    fused: list[dict] = []
    md_sections: list[tuple] = []
    n_model = n_mech = 0

    for ci, chunk in enumerate(chunks, 1):
        cstart = chunk[0]["start"]
        cend = chunk[-1]["end"]
        blks = blocks_in_window(blocks, cstart, cend)
        prompt = build_prompt(chunk, blks)
        resp = ask_doubao(prompt, max_tokens=4096)
        mapping = parse_fusion_response(resp) if resp else None

        if mapping is None:
            method = "机械融合" if not resp else "模型输出解析失败→机械融合"
        else:
            method = "模型"

        chunk_out = []
        for si, s in enumerate(chunk):
            t = ""
            if mapping is not None:
                for item in mapping:
                    if item["i"] == si:
                        t = item["text"]
                        break
            if not t:
                t = mechanical_fusion(s)
                n_mech += 1
            else:
                n_model += 1
            chunk_out.append({"start": s["start"], "end": s["end"], "text": t})

        fused.extend(chunk_out)
        body = "\n".join(f"[{fmt_mmss(s['start'])}] {s['text']}" for s in chunk_out)
        md_sections.append((cstart, cend, body))

        print(f"[ft] 窗 {ci}/{len(chunks)} [{fmt_mmss(cstart)}–{fmt_mmss(cend)}] "
              f"{len(chunk)} 段 × {len(blks)} 块板书 | {method}", file=sys.stderr)

    # ---- 写出 ----
    duration = segs[-1]["end"]
    title = args.title or (args.merged.parent.name if args.merged
                           else args.subtitles.parent.name)

    (args.out_dir / "full_transcript.json").write_text(
        json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8")

    srt_blocks = []
    for i, s in enumerate(fused, 1):
        text = s["text"].replace("\n", " ") if "\n" in s["text"] else s["text"]
        srt_blocks.append(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{text}")
    (args.out_dir / "full_transcript.srt").write_text(
        "\n\n".join(srt_blocks) + "\n", encoding="utf-8")

    meta = f"时间窗 {len(chunks)} | 模型融合 {n_model} 段 | 机械融合 {n_mech} 段"
    md = render_md(title, source, duration, md_sections, meta)
    md = escape_md_dollars(md)
    (args.out_dir / "full_transcript.md").write_text(md, encoding="utf-8")

    print(f"[ok] 完整转写 {len(fused)} 段 -> "
          f"{args.out_dir}/full_transcript.{{json,srt,md}} | {meta}",
          file=sys.stderr)
    print(str(args.out_dir / "full_transcript.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
