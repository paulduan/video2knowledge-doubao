#!/usr/bin/env python3
"""asr_doubao.py — 使用火山引擎 Seed ASR（录音文件识别）进行语音转写。

接口规范见火山引擎官方文档: 录音文件识别标准版
  - 提交: POST https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit
  - 查询: POST https://openspeech.bytedance.com/api/v3/auc/bigmodel/query
  - 认证(新版控制台): X-Api-Key + X-Api-Resource-Id

Pipeline:
    视频 → ffmpeg 提取 16kHz WAV → 上传 TOS → 提交 ASR 任务 → 轮询结果 → 字幕

Usage:
    python3 asr_doubao.py --video in.mp4 --out-dir out --language zh

输出:
    subtitles.srt / subtitles.vtt / subtitles.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

from config import (
    ASR_X_API_KEY,
    ASR_RESOURCE_ID,
    validate_config,
)
from tos_upload import upload_file

# 官方录音文件识别接口地址（新版控制台）
ASR_SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
ASR_QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"


def _build_headers(request_id: str | None = None) -> dict:
    """构建新版控制台认证请求头。"""
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": ASR_X_API_KEY,
        "X-Api-Resource-Id": ASR_RESOURCE_ID,
    }
    if request_id:
        headers["X-Api-Request-Id"] = request_id
    return headers


def extract_wav(video: Path, out_dir: Path) -> Path:
    """从视频提取 16kHz 单声道 WAV。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio_16k.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", str(wav)],
        check=True,
    )
    return wav


def get_audio_duration(wav: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def submit_asr_task(audio_url: str, language: str = "zh-CN") -> str:
    """提交异步文件转写任务，返回 task_id（即 X-Api-Request-Id）。"""
    request_id = uuid.uuid4().hex
    payload = {
        "user": {"uid": "video2knowledge"},
        "audio": {
            "url": audio_url,
            "format": "wav",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
            "language": language,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "show_utterances": True,
        },
    }
    resp = requests.post(
        ASR_SUBMIT_URL,
        headers=_build_headers(request_id),
        json=payload,
        timeout=60,
    )

    # 提交成功返回 HTTP 200 空 body；状态通过 header X-Api-Status-Code 判断
    status_code = resp.headers.get("X-Api-Status-Code", "")
    if resp.status_code != 200 or status_code != "20000000":
        msg = resp.headers.get("X-Api-Message", "") or resp.text
        raise RuntimeError(f"ASR 提交失败: {msg}")

    return request_id


def query_asr_result(task_id: str, max_wait: int = 600,
                     poll_interval: int = 5) -> list[dict]:
    """轮询获取识别结果，返回带时间戳的分句数组。

    用相同的 X-Api-Request-Id 轮询查询接口：
      - 20000000 成功
      - 20000001/20000002 处理中/排队中，继续轮询
      - 其他为失败
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        resp = requests.post(
            ASR_QUERY_URL,
            headers=_build_headers(task_id),
            json={},
            timeout=60,
        )
        status_code = resp.headers.get("X-Api-Status-Code", "")
        msg = resp.headers.get("X-Api-Message", "") or ""

        # 仍在处理中/排队中 → 继续轮询
        if status_code in ("20000001", "20000002"):
            elapsed = int(time.time() - start_time)
            print(f"[asr] 等待中... ({elapsed}s, {msg})", file=sys.stderr)
            time.sleep(poll_interval)
            continue

        if status_code != "20000000":
            raise RuntimeError(f"ASR 查询失败: {msg}")

        try:
            data = resp.json()
        except ValueError:
            data = {}

        result = data.get("result")
        if result and isinstance(result, dict) and result.get("text"):
            return result.get("utterances") or []

        elapsed = int(time.time() - start_time)
        print(f"[asr] 等待中... ({elapsed}s)", file=sys.stderr)
        time.sleep(poll_interval)

    raise TimeoutError(f"ASR 任务超时（{max_wait}s）: task_id={task_id}")


def parse_asr_result(utterances: list) -> list[dict]:
    """解析 ASR 转写结果为 [{start, end, text}]。

    utterances 数组中每个元素包含 text / start_time / end_time，
    时间单位为毫秒。
    """
    segments = []
    if not utterances:
        return segments

    for item in utterances:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        start_ms = item.get("start_time", 0)
        end_ms = item.get("end_time", start_ms + 1000)
        segments.append({
            "start": round(start_ms / 1000.0, 3),
            "end": round(end_ms / 1000.0, 3),
            "text": text,
        })

    return segments


def fmt_ts(sec: float, sep: str = ",") -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segs: list[dict]) -> str:
    out = []
    for i, s in enumerate(segs, 1):
        out.append(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text']}\n")
    return "\n".join(out)


def to_vtt(segs: list[dict]) -> str:
    body = "\n".join(
        f"{fmt_ts(s['start'], sep='.')} --> {fmt_ts(s['end'], sep='.')}\n{s['text']}\n"
        for s in segs
    )
    return "WEBVTT\n\n" + body


def main() -> int:
    ap = argparse.ArgumentParser(description="火山引擎 ASR → 时间戳字幕")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--language", default="zh-CN",
                    help="语言代码 (zh-CN/en-US/ja-JP 等, 默认 zh-CN)")
    ap.add_argument("--max-wait", type=int, default=600,
                    help="最大等待秒数 (默认 600)")
    ap.add_argument("--poll-interval", type=int, default=5,
                    help="轮询间隔秒数 (默认 5)")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] 视频不存在: {args.video}", file=sys.stderr)
        return 2

    try:
        validate_config(service="asr")
    except RuntimeError as e:
        print(f"[err] {e}", file=sys.stderr)
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 提取音频
    print("[asr] 提取 16kHz 单声道音频...", file=sys.stderr)
    wav = extract_wav(args.video, args.out_dir)
    duration = get_audio_duration(wav)
    print(f"[asr] 音频时长: {duration:.1f}s", file=sys.stderr)

    # 2. 上传 TOS
    print("[asr] 上传音频到 TOS...", file=sys.stderr)
    audio_url = upload_file(wav)
    print("[asr] TOS 上传完成", file=sys.stderr)

    # 3. 提交 ASR 任务
    print("[asr] 提交识别任务...", file=sys.stderr)
    task_id = submit_asr_task(audio_url, language=args.language)
    print(f"[asr] task_id={task_id}", file=sys.stderr)

    # 4. 轮询结果
    print("[asr] 等待识别完成...", file=sys.stderr)
    transcriptions = query_asr_result(task_id, args.max_wait, args.poll_interval)

    # 5. 解析输出
    segments = parse_asr_result(transcriptions)
    print(f"[asr] 识别完成: {len(segments)} 个语句", file=sys.stderr)

    (args.out_dir / "subtitles.srt").write_text(to_srt(segments), encoding="utf-8")
    (args.out_dir / "subtitles.vtt").write_text(to_vtt(segments), encoding="utf-8")
    (args.out_dir / "subtitles.json").write_text(
        json.dumps({"language": args.language, "duration": duration,
                    "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"[ok] {len(segments)} 段字幕 -> {args.out_dir}/subtitles.{{srt,vtt,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
