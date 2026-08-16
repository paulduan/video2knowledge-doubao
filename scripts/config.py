#!/usr/bin/env python3
"""config.py — 火山引擎 / 豆包 API 统一配置。

从 .env 加载配置，提供全局常量。
"""
from __future__ import annotations

import os
from pathlib import Path

import dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(PROJECT_ROOT / ".env")

# 火山引擎全局密钥
VOLC_ACCESS_KEY = os.environ.get("VOLC_ACCESS_KEY", "")
VOLC_SECRET_KEY = os.environ.get("VOLC_SECRET_KEY", "")

# TOS 对象存储
TOS_ENDPOINT = os.environ.get("TOS_ENDPOINT", "tos-cn-beijing.volces.com")
TOS_BUCKET = os.environ.get("TOS_BUCKET", "")
TOS_REGION = os.environ.get("TOS_REGION", "cn-beijing")

# 豆包 ASR（语音识别）— 录音文件识别标准版
# 说明: 当前账号开通的是 1.0 (volc.bigasr.auc)。若控制台已开通 2.0 (volc.seedasr.auc) 可改回
ASR_X_API_KEY = os.environ.get("ASR_X_API_KEY", "")
ASR_RESOURCE_ID = os.environ.get("ASR_RESOURCE_ID", "volc.bigasr.auc")

# 火山方舟（Ark）— 视觉模型 & 文本模型共用
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_EP_ID = os.environ.get("ARK_EP_ID", "")  # doubao-seed-2-1-pro 视觉模型接入点
ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ARK_URL = os.environ.get("ARK_URL", f"{ARK_BASE_URL}/chat/completions")

# 文本模型（知识生成用，可以与视觉模型共用或使用独立接入点）
ARK_TEXT_EP_ID = os.environ.get("ARK_TEXT_EP_ID", "")  # 纯文本模型接入点（可选）

# ffmpeg / ffprobe 可执行文件
# 优先使用项目 bin/ 目录（自动软链到 imageio-ffmpeg 自带的二进制），
# 其次使用系统 PATH 中的命令。
BIN_DIR = PROJECT_ROOT / "bin"


def _add_bin_to_path() -> None:
    """把项目 bin/ 目录加入 PATH（如果里面存在 ffmpeg）。"""
    if BIN_DIR.is_dir() and any(f in os.listdir(BIN_DIR) for f in ("ffmpeg", "ffprobe")):
        if str(BIN_DIR) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{BIN_DIR}:{os.environ.get('PATH', '')}"


_add_bin_to_path()


def validate_config(service: str = "all") -> None:
    """校验必要配置是否已设置。"""
    missing = []

    if not VOLC_ACCESS_KEY:
        missing.append("VOLC_ACCESS_KEY")
    if not VOLC_SECRET_KEY:
        missing.append("VOLC_SECRET_KEY")

    if service in ("asr", "all"):
        if not ASR_X_API_KEY:
            missing.append("ASR_X_API_KEY")
        if not TOS_BUCKET:
            missing.append("TOS_BUCKET")

    if service in ("ocr", "vision", "all"):
        if not ARK_API_KEY:
            missing.append("ARK_API_KEY")
        if not ARK_EP_ID:
            missing.append("ARK_EP_ID")

    if service in ("knowledge", "all"):
        if not ARK_API_KEY:
            missing.append("ARK_API_KEY")

    if missing:
        raise RuntimeError(
            f"缺少必要配置: {', '.join(missing)}。请在 .env 文件中设置。"
        )
