#!/usr/bin/env python3
"""build_full_transcript.py — 生成包含全部课程内容的完整文档（语音 + 板书）。

与 build_knowledge.py（豆包大模型总结的知识文档）互补：
本脚本纯本地处理，把 ASR 全部语音转写与 OCR 全部板书按时间线完整输出，
不调用任何付费 API。

输出（写入 --out-dir）:
  - full_transcript.md — Markdown 完整课程文档

文档结构:
  头部  标题 + 元信息（时长、ASR 段数、板书块数、生成时间）
  正文  以语音为连续主时间线，按时间顺序输出全部语音段落
        （时间上连续的片段合并为自然段，每 5 分钟分节）；
        当时间线到达某块板书的时间段时，将板书内容内联插入到语音流中
        （每块板书只在其首次出现处插入一次）。

若未提供 --captions，则只生成语音部分（不插入板书）。

Usage:
    python3 build_full_transcript.py \
        --subtitles output/xxx/subtitles.json \
        --captions  output/xxx/captions.json \
        --out-dir   output/xxx
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import config  # noqa: F401  复用 config.py 的 PATH 注入（bin/ 目录），保持项目脚本风格一致

# 语音片段合并阈值（秒）：相邻两条语音间隔 <= 该值视为同一句话的延续，合并为自然段
MERGE_GAP = 1.5

# 正文分节步长（秒）：按此时间跨度切分子标题，便于长视频跳转阅读
SECTION_STEP = 300  # 5 分钟


def load_json(path: Path) -> dict | list:
    """读取 UTF-8 编码的 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def load_segments(path: Path) -> list[dict]:
    """加载 {start, end, text} 列表，兼容 subtitles.json / captions.json / merged.json。"""
    data = load_json(path)
    if isinstance(data, list):
        return data
    return data.get("segments") or data.get("captions") or []


def fmt_ts(sec: float) -> str:
    """秒数格式化为 MM:SS；超过 1 小时用 H:MM:SS。"""
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _join_text(a: str, b: str) -> str:
    """拼接相邻语音文本：中文之间不加空格；英文/数字相邻时补一个空格。"""
    if not a:
        return b
    if not b:
        return a
    if a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
        return f"{a} {b}"
    return a + b


def merge_paragraphs(segments: list[dict], gap: float = MERGE_GAP) -> list[dict]:
    """把时间上连续的语音片段合并为自然段落。

    段落保留起始/结束时间；相邻片段间隔 <= gap 秒视为同一句话的延续。
    """
    paras: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg["start"]
        end = seg["end"]
        if paras and start - paras[-1]["end"] <= gap:
            last = paras[-1]
            last["end"] = max(last["end"], end)
            last["text"] = _join_text(last["text"], text)
        else:
            paras.append({"start": start, "end": end, "text": text})
    return paras


def render_header(title: str, duration: float, asr_count: int,
                  board_count: int, para_count: int, source: str) -> str:
    """文档头部：标题 + 元信息。"""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {title}",
        "",
        "> 完整课程文档（语音 + 板书）",
        f"> 语音来源: {source}",
        f"> 视频时长: {fmt_ts(duration)} | 语音段: {asr_count} 段"
        f"（合并为 {para_count} 个自然段） | 板书块: {board_count} 块",
        f"> 生成时间: {now}",
    ]
    return "\n".join(lines)


def render_body(paras: list[dict], boards: list[dict], duration: float) -> str:
    """正文：以语音为连续主时间线，在到达板书时间段时内联插入板书内容。

    - 语音自然段按 start 时间顺序输出（每 5 分钟一节）；
    - 处理到某段语音前，先输出所有 start 已到达、尚未插入的板书块
      （板书按其自身时间点内联插入，每块只插入一次）；
    - 若某块板书的 start 晚于最后一段语音，则在文档末尾补插。
    """
    lines = [
        "## 完整语音转写（板书内联插入）",
        "",
        "> 说明: 以语音为连续主时间线；当时间到达某块板书的时间段时，",
        "> 将板书内容内联插入到语音流中（每块板书只在其首次出现处插入一次）。",
        f"> 共 {len(paras)} 个自然段、{len(boards)} 块板书。",
        "",
    ]
    board_events = [(b["start"], (b.get("text") or "").rstrip())
                    for b in boards if (b.get("text") or "").strip()]
    bi = 0
    cur_sec: int | None = None

    def section(sec: int) -> None:
        """输出（若尚未输出过）第 sec 秒所在分节的标题。"""
        nonlocal cur_sec
        if sec == cur_sec:
            return
        cur_sec = sec
        end = min(sec + SECTION_STEP, int(duration))
        lines.append(f"### {fmt_ts(sec)} – {fmt_ts(end)}")
        lines.append("")

    def emit_boards(before: float) -> None:
        """输出所有 start 已到达（<= before）且尚未插入的板书块。"""
        nonlocal bi
        while bi < len(board_events) and board_events[bi][0] <= before:
            start, text = board_events[bi]
            bi += 1
            section(int(start) // SECTION_STEP * SECTION_STEP)
            lines.extend([f"【板书 @{fmt_ts(start)}】", "",
                          "```text", text, "```", ""])

    for p in paras:
        emit_boards(p["start"])
        section(int(p["start"]) // SECTION_STEP * SECTION_STEP)
        lines.append(f"[{fmt_ts(p['start'])}] {p['text']}")
        lines.append("")
    emit_boards(float("inf"))
    return "\n".join(lines).rstrip() + "\n"


def escape_md_dollars(text: str) -> str:
    """转义 Markdown 中未被转义的 $（OCR 板书 LaTeX 公式定界符），避免渲染异常。

    只转义前面没有反斜杠的独立 $；已转义的 \\$ 保持原样，防止二次转义。
    """
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成包含全部课程内容的完整文档（语音 + 板书，纯本地处理）")
    ap.add_argument("--subtitles", required=True, type=Path,
                    help="ASR 语音转写 subtitles.json")
    ap.add_argument("--captions", type=Path, default=None,
                    help="OCR 板书 captions.json（可选，给出则内联插入板书）")
    ap.add_argument("--out-dir", required=True, type=Path, help="输出目录")
    ap.add_argument("--title", default=None, help="文档标题（默认取字幕所在目录名）")
    args = ap.parse_args()

    if not args.subtitles.is_file():
        print(f"[err] 字幕文件不存在: {args.subtitles}", file=sys.stderr)
        return 2
    if args.captions and not args.captions.is_file():
        print(f"[err] 板书文件不存在: {args.captions}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 ASR 语音转写（默认来自 subtitles.json）
    sub = load_json(args.subtitles)
    asr = load_segments(args.subtitles)
    if not asr:
        print(f"[err] 无 ASR 片段: {args.subtitles}", file=sys.stderr)
        return 2

    # 加载板书（可选）：优先 merged.json 的 visual_blocks（内容与 captions.json 一致），
    # 其次 captions.json 本身。
    boards: list[dict] = []
    source = args.subtitles.name
    if args.captions:
        merged_path = args.subtitles.parent / "merged.json"
        if merged_path.is_file():
            merged = load_json(merged_path)
            m_segments = merged.get("segments") or []
            if m_segments:
                # 融合模式的 segments 语音完整（476 段），且每条语音段直接绑定板书
                asr = m_segments
                source = "merged.json（融合模式，含板书绑定）"
                print(f"[info] 使用 {merged_path.name} 作为语音来源 "
                      f"（{len(asr)} 段）", file=sys.stderr)
            m_blocks = merged.get("visual_blocks") or []
            if m_blocks:
                boards = m_blocks
                print(f"[info] 板书来自 {merged_path.name}.visual_blocks "
                      f"（{len(boards)} 块）", file=sys.stderr)
        if not boards:
            boards = load_segments(args.captions)

    # 时长：优先 subtitles.json 的 duration 字段，缺失时取最后一段的结束时间
    duration = sub.get("duration") if isinstance(sub, dict) else None
    if not duration:
        duration = asr[-1]["end"]

    # 标题：--title 优先，否则取字幕所在目录名（即视频标题）
    title = args.title or args.subtitles.parent.name

    paras = merge_paragraphs(asr)

    # 组装 Markdown：头部 + 正文（语音主流 + 板书内联）
    md = (render_header(title, duration, len(asr), len(boards), len(paras), source)
          + "\n\n" + render_body(paras, boards, duration))
    md = escape_md_dollars(md)

    md_path = args.out_dir / "full_transcript.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[ok] 完整课程文档 -> {md_path}", file=sys.stderr)

    total_lines = md.count("\n") + 1
    print(f"[info] {len(asr)} 段语音 | {len(boards)} 块板书 | "
          f"合并为 {len(paras)} 个自然段 | md 共 {total_lines} 行", file=sys.stderr)
    print(str(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
