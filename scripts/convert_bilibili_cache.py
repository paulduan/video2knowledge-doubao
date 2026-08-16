#!/usr/bin/env python3
"""convert_bilibili_cache.py — 将 B 站客户端缓存的 m4s 音视频合并封装为 mp4。

B 站桌面/移动端缓存的目录形如:
  <cache_dir>/<cid>/<cid>-1-<videoType>.m4s   # 视频流 (H.264/HEVC)
  <cache_dir>/<cid>/<cid>-1-<audioType>.m4s   # 音频流 (AAC)
  <cache_dir>/<cid>/videoInfo.json       # 元数据 (标题/时长/清晰度等)

缓存的 .m4s 文件头部带有 9 字节占位前缀 ("000000000" + NUL)，ffmpeg 无法
直接解析；本脚本会跳过该前缀后再交给 ffmpeg，用 -c copy 无损封装为 mp4，
不重新编码，速度很快。

Usage:
    # 列出缓存目录下所有可用的缓存（含未下载的占位项）
    python3 convert_bilibili_cache.py --list

    # 转换指定缓存 (cid) 到 output/
    python3 convert_bilibili_cache.py --cid 25685856471

    # 转换指定缓存到自定义目录
    python3 convert_bilibili_cache.py --cid 25685856471 --out-dir out

    # 批量转换所有完整缓存
    python3 convert_bilibili_cache.py --all --out-dir out
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# 导入 config 把项目 bin/ 加入 PATH（ffmpeg）
from config import BIN_DIR  # noqa: F401  (副作用: 注册 bin 到 PATH)

DEFAULT_CACHE_DIR = Path("/Users/duanp/Movies/bilibili")

# B 站缓存 m4s 头部有 9 字节的 ASCII '0' 占位前缀，
# 紧随其后的才是标准 ISO-BMFF 内容（box size 首字节恰为 0x00）
_M4S_HEADER = b"000000000"


def _safe_filename(name: str) -> str:
    """把标题转成文件系统安全的文件名。"""
    cleaned = re.sub(r'[/\\:*?"<>|]', "_", name).strip()
    if not cleaned or cleaned in (".", ".."):
        return "video"
    return cleaned


def _read_video_info(cache_dir: Path) -> dict:
    """读取 videoInfo.json，返回元数据（不存在时返回空 dict）。"""
    for name in ("videoInfo.json", ".videoInfo", "entry.json"):
        p = cache_dir / name
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def scan_caches(cache_dir: Path) -> list[dict]:
    """扫描缓存目录，返回每个子目录的信息列表。"""
    if not cache_dir.is_dir():
        raise RuntimeError(f"缓存目录不存在: {cache_dir}")
    caches = []
    for d in sorted(cache_dir.iterdir()):
        if not d.is_dir():
            continue
        info = _read_video_info(d)
        m4s = [f for f in d.iterdir() if f.suffix == ".m4s" and f.is_file()]
        caches.append({
            "cid": d.name,
            "title": info.get("title") or d.name,
            "duration": info.get("duration") or 0,
            "status": "ok" if len(m4s) >= 2 else "no-video",
            "m4s": [f.name for f in m4s],
        })
    return caches


def _strip_m4s_header(src: Path, dst: Path) -> None:
    """把 m4s 复制到 dst 并去掉头部占位前缀（无前缀则原样复制）。"""
    offset = 0
    with src.open("rb") as f:
        if f.read(len(_M4S_HEADER)) == _M4S_HEADER:
            offset = len(_M4S_HEADER)
    with src.open("rb") as f, dst.open("wb") as o:
        f.seek(offset)
        shutil.copyfileobj(f, o, 1024 * 1024)


def _probe_kind(path: Path) -> str:
    """探测去掉头部后的流类型: video / audio / unknown。"""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)],
        capture_output=True, text=True,
    )
    err = r.stderr
    if "Video:" in err:
        return "video"
    if "Audio:" in err:
        return "audio"
    return "unknown"


def convert(cache_dir: Path, out_dir: Path, force: bool = False) -> Path:
    """把单个缓存目录合并封装为 mp4，返回生成的 mp4 路径。"""
    cache_dir = cache_dir.resolve()
    if not cache_dir.is_dir():
        raise RuntimeError(f"缓存目录不存在: {cache_dir}")
    info = _read_video_info(cache_dir)
    title = _safe_filename(info.get("title") or cache_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    m4s_files = sorted(f for f in cache_dir.iterdir()
                       if f.suffix == ".m4s" and f.is_file())
    if len(m4s_files) < 2:
        raise RuntimeError(f"{cache_dir.name}: 缓存不完整，缺少视频/音频流 "
                           f"(仅 {len(m4s_files)} 个 m4s)")

    # 探测每个流的类型，构造去头临时文件
    with tempfile.TemporaryDirectory(prefix="bili_cache_") as td:
        tmp = Path(td)
        streams: dict[str, Path | None] = {"video": None, "audio": None}
        for f in m4s_files:
            stripped = tmp / f"{f.name}.strip.m4s"
            _strip_m4s_header(f, stripped)
            kind = _probe_kind(stripped)
            if kind in streams and streams[kind] is None:
                streams[kind] = stripped
            else:
                stripped.unlink(missing_ok=True)
        if streams["video"] is None or streams["audio"] is None:
            raise RuntimeError(f"{cache_dir.name}: 无法识别视频/音频流 "
                               f"({m4s_files})")

        # 文件名: 标题.mp4；存在且未强制覆盖时加序号
        out = out_dir / f"{title}.mp4"
        n = 2
        while out.exists() and not force:
            out = out_dir / f"{title}_{n}.mp4"
            n += 1

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(streams["video"]),
            "-i", str(streams["audio"]),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c", "copy", "-movflags", "+faststart",
            str(out),
        ]
        subprocess.run(cmd, check=True)

    if not out.is_file():
        raise RuntimeError(f"{cache_dir.name}: 合并失败，未生成 {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="B 站缓存 m4s → mp4")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help=f"缓存根目录 (默认 {DEFAULT_CACHE_DIR})")
    ap.add_argument("--cid", help="指定转换的缓存目录名 (cid)")
    ap.add_argument("--all", action="store_true", help="批量转换所有完整缓存")
    ap.add_argument("--list", action="store_true", help="列出所有缓存")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="mp4 输出目录 (默认使用 --cache-dir 上级目录的 "
                         "output/converted/)")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的同名 mp4")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = args.cache_dir.parent / "converted"

    try:
        if args.list:
            caches = scan_caches(args.cache_dir)
            if not caches:
                print("[ok] 缓存目录为空", file=sys.stderr)
                return 0
            print(f"{'CID':<14} {'时长':>6}  {'状态':<10} 标题")
            for c in caches:
                dur = f"{c['duration'] // 60}m" if c["duration"] else "  -"
                print(f"{c['cid']:<14} {dur:>6}  {c['status']:<10} {c['title']}")
            ok = sum(1 for c in caches if c["status"] == "ok")
            print(f"\n共 {len(caches)} 个缓存，其中 {ok} 个可转换 "
                  f"(视频+音频完整)", file=sys.stderr)
            return 0

        if args.all:
            caches = [c for c in scan_caches(args.cache_dir)
                      if c["status"] == "ok"]
            if not caches:
                print("[err] 没有完整缓存可转换", file=sys.stderr)
                return 1
            failed = 0
            for c in caches:
                try:
                    out = convert(args.cache_dir / c["cid"], args.out_dir,
                                  args.force)
                    print(f"[ok] {c['cid']} -> {out}")
                except Exception as e:
                    failed += 1
                    print(f"[err] {c['cid']}: {e}", file=sys.stderr)
            if failed:
                print(f"[err] {failed}/{len(caches)} 个转换失败",
                      file=sys.stderr)
                return 1
            print(f"[ok] 完成，共 {len(caches)} 个 -> {args.out_dir}",
                  file=sys.stderr)
            return 0

        if not args.cid:
            ap.error("必须指定 --cid、--all 或 --list")
        try:
            out = convert(args.cache_dir / args.cid, args.out_dir, args.force)
        except Exception as e:
            print(f"[err] 转换失败: {e}", file=sys.stderr)
            return 1
        print(f"[ok] 已转换 -> {out}")
        print(str(out))
        return 0

    except Exception as e:
        print(f"[err] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
