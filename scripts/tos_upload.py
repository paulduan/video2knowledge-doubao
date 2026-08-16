#!/usr/bin/env python3
"""tos_upload.py — 火山引擎 TOS 对象存储上传工具。

ASR 音频 / OCR 图片通过 TOS 预签名 URL 提交给豆包/火山服务。
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path

import tos

from config import (
    VOLC_ACCESS_KEY,
    VOLC_SECRET_KEY,
    TOS_ENDPOINT,
    TOS_BUCKET,
    TOS_REGION,
)

_client: tos.TosClientV2 | None = None


def _get_client() -> tos.TosClientV2:
    global _client
    if _client is None:
        _client = tos.TosClientV2(
            ak=VOLC_ACCESS_KEY,
            sk=VOLC_SECRET_KEY,
            endpoint=TOS_ENDPOINT,
            region=TOS_REGION,
        )
    return _client


def upload_file(local_path: str | Path, object_key: str | None = None,
                expires_hours: int = 12) -> str:
    """上传本地文件到 TOS，返回预签名下载 URL。

    Args:
        local_path: 本地文件路径
        object_key: TOS 对象 key（默认自动生成 v2k/<uuid>/<filename>）
        expires_hours: URL 有效时长（小时）

    Returns:
        预签名 HTTPS URL
    """
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"文件不存在: {local_path}")

    if object_key is None:
        run_id = uuid.uuid4().hex[:12]
        object_key = f"v2k/{run_id}/{local_path.name}"

    client = _get_client()
    client.put_object_from_file(TOS_BUCKET, object_key, str(local_path))

    url = client.pre_signed_url(
        tos.HttpMethodType.Http_Method_Get,
        TOS_BUCKET,
        object_key,
        expires=int(timedelta(hours=expires_hours).total_seconds()),
    )
    return url.signed_url


def cleanup_object(object_key: str) -> None:
    """删除 TOS 上的临时对象。"""
    try:
        client = _get_client()
        client.delete_object(TOS_BUCKET, object_key)
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 tos_upload.py <local_file> [object_key]")
        sys.exit(1)
    url = upload_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"预签名 URL: {url}")
