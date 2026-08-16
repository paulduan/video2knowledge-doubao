#!/usr/bin/env python3
"""build_knowledge.py — 将字幕/OCR 结果精炼为知识文档。

消费 SRT/JSON 字幕（任一路径的输出）并生成:
  - knowledge.md   — 结构化知识文档（模板渲染）
  - cards.csv      — 问答闪卡

摘要/问答/术语提取交给豆包大模型 API，整条流水线保持一致的 API 来源。

Usage:
    python3 build_knowledge.py --subtitles out/subtitles.json --out-dir out
"""
from __future__ import annotations

import argparse
import csv
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

DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "default-template.md"

# 知识生成优先使用文本模型接入点，如果没有则复用视觉模型接入点
_TEXT_MODEL = ARK_TEXT_EP_ID or ARK_EP_ID


def load_subtitles(path: Path) -> tuple[list[dict], str]:
    """加载字幕。支持 JSON 和 SRT 格式。"""
    txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(txt)
        if isinstance(data, list):
            return data, path.name
        segs = data.get("segments") or data.get("captions") or []
        return segs, path.name
    # SRT 解析
    segs = []
    for block in re.split(r"\n\s*\n", txt.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(
            r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)",
            lines[1],
        )
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        segs.append({"start": start, "end": end, "text": " ".join(lines[2:])})
    return segs, path.name


def fmt_mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def load_merged(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_interleaved_text(segs: list[dict]) -> str:
    """构建音频+画面交织文本（供 LLM 理解）。"""
    seen_visual: set[int] = set()
    lines = []
    for s in segs:
        ts = fmt_mmss(s["start"])
        audio = s["text"].strip()
        vis = (s.get("visual") or "").strip()
        if vis:
            vid = hash(vis)
            if vid in seen_visual:
                lines.append(f"[{ts}] 🎙️{audio}  | 🖼️(画面同上)")
            else:
                seen_visual.add(vid)
                lines.append(f"[{ts}] 🎙️{audio}\n🖼️画面:\n{vis}")
        else:
            lines.append(f"[{ts}] 🎙️{audio}")
    return "\n".join(lines)


def ask_doubao(prompt: str, system: str = "你是一个视频内容分析助手。") -> str | None:
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
        "max_tokens": 2000,
        # doubao-seed-2-1-pro 默认深度思考，会导致响应耗时极长。
        # 知识生成是批处理场景，显式禁用 thinking 以保障速度。
        "thinking": {"type": "disabled"},
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ARK_API_KEY}",
    }

    req = urllib.request.Request(ARK_URL, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return None
    except (urllib.error.URLError, OSError):
        return None


def _chunk_lines(raw_text: str, max_chars: int) -> list[str]:
    """将文本按行分割为不超过 max_chars 的块。"""
    lines = raw_text.splitlines()
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        n = len(ln) + 1
        if cur and cur_len + n > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks or [raw_text[:max_chars]]


def _strip_fence(resp: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\s*\n?", "", resp.strip())
    return re.sub(r"\n?```\s*$", "", cleaned).strip()


def _merge_list_items(items: list[str], max_n: int) -> str:
    """去重并限制列表条数。"""
    seen, out = set(), []
    for it in items:
        key = re.sub(r"\s+", "", it).strip("- ")
        if key and key not in seen:
            seen.add(key)
            out.append(it)
            if len(out) >= max_n:
                break
    return "\n".join(out)


def build_analysis(raw_text: str, char_limit: int = 8000) -> dict:
    """分字段向豆包大模型请求摘要/时间线/要点/问答/术语。

    策略: 每个字段单独一次 prompt，小而精准。长文本使用 map-reduce。
    """
    fields = {
        "summary": "", "timeline": "", "key_points": "",
        "qa": "", "glossary": "",
    }

    sub = raw_text[:char_limit]
    CHUNK = 4500
    long_mode = len(sub) > CHUNK * 1.5
    chunks = _chunk_lines(sub, CHUNK) if long_mode else [sub]

    TASKS = [
        ("summary",
         "任务：读懂下面这段视频字幕，用中文写3-5句话总结核心内容与目的。\n"
         "要求：用自己的话概括，禁止照抄原句，只输出总结正文。"),
        ("timeline",
         "任务：从视频字幕中提炼最多8个关键事件节点。\n"
         "格式：`- [mm:ss] 概括性事件描述(不超过15字)`\n只输出列表。"),
        ("key_points",
         "任务：从视频字幕中提炼最多8个核心知识点。\n"
         "格式：`- 知识点(一句话概括)`\n只输出列表。"),
        ("qa",
         "任务：基于视频字幕设计6-10组中文问答。\n"
         "格式：每组两行 `Q: 问题` 和 `A: 答案`\n"
         "问答必须基于字幕内容，禁止编造。只输出问答。"),
        ("glossary",
         "任务：从视频字幕中提取重要术语/专有名词。\n"
         "格式：`- 术语`\n只保留名词性术语，只输出列表。"),
    ]

    LIST_FIELDS = {"timeline", "key_points", "qa", "glossary"}
    CAPS = {"timeline": 12, "key_points": 12, "qa": 30, "glossary": 16}

    for key, instruction in TASKS:
        try:
            if long_mode and key in LIST_FIELDS:
                collected = []
                for ch in chunks:
                    resp = ask_doubao(instruction + f"\n\n字幕:\n{ch}")
                    if resp and len(resp.strip()) > 3:
                        collected.extend(
                            l for l in _strip_fence(resp).splitlines() if l.strip()
                        )
                fields[key] = _merge_list_items(collected, CAPS[key])
            elif long_mode and key == "summary":
                parts = []
                for ch in chunks:
                    resp = ask_doubao(instruction + f"\n\n字幕:\n{ch}")
                    if resp and len(resp.strip()) > 3:
                        parts.append(_strip_fence(resp))
                joined = "\n".join(parts)
                reduce_instr = (
                    "任务：下面是视频各部分的摘要，请融合成3-5句话的总体总结，"
                    "保留关键信息，去掉重复，只输出总结正文。"
                )
                resp = ask_doubao(reduce_instr + f"\n\n摘要:\n{joined[:6000]}")
                fields[key] = _strip_fence(resp) if resp else joined[:500]
            else:
                resp = ask_doubao(instruction + f"\n\n字幕:\n{sub}")
                fields[key] = _strip_fence(resp) if resp and len(resp.strip()) > 3 else ""
        except Exception:
            pass

    # 填充空字段
    n = raw_text.count("\n") + 1
    defaults = {
        "summary": f"(模型不可用，原始字幕共 {n} 行)",
        "timeline": "- [未分段] " + (raw_text.splitlines()[0][:60] if raw_text else ""),
        "key_points": "- 原始字幕见下方；配置豆包 API 可生成结构化要点",
        "qa": "Q: (配置豆包 API 后自动生成问答)\nA: ...",
        "glossary": "- (配置豆包 API 后自动生成术语表)",
    }
    for k in fields:
        if not fields[k]:
            fields[k] = defaults.get(k, "")
    return fields


def render_template(template_path: Path, ctx: dict) -> str:
    tpl = template_path.read_text(encoding="utf-8")
    for k, v in ctx.items():
        tpl = tpl.replace("{{" + k + "}}", str(v))
    return tpl


def escape_md_dollars(text: str) -> str:
    """转义 Markdown 中未被转义的 $（OCR 板书 LaTeX 公式定界符），避免渲染异常。

    只转义前面没有反斜杠的独立 $；已转义的 \\$ 保持原样，防止二次转义。
    """
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def cards_from_qa(qa: str, source: str) -> list[list[str]]:
    """解析 Q/A 文本为 CSV 行。"""
    rows: list[list[str]] = []
    if not isinstance(qa, str):
        return rows
    cur_q, cur_a = None, None
    for ln in qa.splitlines():
        s = ln.strip()
        if s.lower().startswith("q:"):
            if cur_q is not None:
                rows.append([cur_q, (cur_a or "").strip(), "", "", source])
            cur_q, cur_a = s[2:].strip(), None
        elif s.lower().startswith("a:"):
            cur_a = s[2:].strip()
    if cur_q is not None:
        rows.append([cur_q, (cur_a or "").strip(), "", "", source])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="字幕 → 知识文档")
    ap.add_argument("--subtitles", required=True, type=Path)
    ap.add_argument("--merged", type=Path, default=None,
                    help="merged.json (融合模式)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--format", choices=["knowledge", "csv", "all"],
                    default="all", help="输出格式: knowledge=md, csv=闪卡, "
                                        "all=两者 (默认 all)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--char-limit", type=int, default=None)
    args = ap.parse_args()

    if not args.subtitles.is_file():
        print(f"[err] 字幕文件不存在: {args.subtitles}", file=sys.stderr)
        return 2
    if not args.template.is_file():
        print(f"[err] 模板不存在: {args.template}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.merged and args.merged.is_file():
        merged_data = load_merged(args.merged)
        mseg = merged_data["segments"]
        raw_text = build_interleaved_text(mseg)
        char_limit = args.char_limit or 20000
        segs = mseg
        source = f"{args.merged.name} (融合模式)"
    else:
        segs, source = load_subtitles(args.subtitles)
        raw_text = "\n".join(f"[{fmt_mmss(s['start'])}] {s['text']}" for s in segs)
        char_limit = args.char_limit or 8000

    title = args.title or args.subtitles.stem.replace("_", " ")
    duration = segs[-1]["end"] if segs else 0.0
    print(f"[v2k] {len(segs)} 段, {duration:.0f}s; 正在生成知识文档...",
          file=sys.stderr)

    analysis = build_analysis(raw_text, char_limit=char_limit)

    ctx = {
        "title": title,
        "source": source,
        "duration": fmt_mmss(duration),
        "date": dt.date.today().isoformat(),
        "summary": analysis["summary"],
        "timeline": analysis["timeline"],
        "key_points": analysis["key_points"],
        "qa": analysis["qa"],
        "glossary": analysis["glossary"],
        "visual_timeline": "(无视觉信息)" if not args.merged else "",
        "meta": f"segments={len(segs)} model={_TEXT_MODEL}",
    }

    want = {"knowledge", "csv"} if args.format == "all" else {args.format}

    if "knowledge" in want:
        md = render_template(args.template, ctx)
        md = escape_md_dollars(md)
        md_path = args.out_dir / "knowledge.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"[ok] knowledge doc -> {md_path}")

    if "csv" in want:
        rows = cards_from_qa(analysis["qa"], source)
        csv_path = args.out_dir / "cards.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "answer", "tags", "timestamp", "source"])
            w.writerows(rows)
        print(f"[ok] {len(rows)} 张闪卡 -> {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
