"""Feishu (Lark) IM helpers built on the official ``lark-oapi`` SDK.

This module exposes a small, focused API around ``lark.Client.im.v1`` that the
rest of the project uses:

- :func:`reply_text` / :func:`reply_image` — reply to a specific user message
  (preferred in groups for threaded responses).
- :func:`send_text` / :func:`send_image` — send a fresh message into a chat
  by ``chat_id`` (used by the one-shot ``run`` CLI).
- :func:`download_message_resource` — download an image attached to a
  received message.
- :func:`get_message` — fetch a single message by id (used to resolve
  quoted-message images).
- :func:`list_chat_messages` — fetch the recent message history of a chat
  (used by the bot to build conversation context).
- :func:`upload_image` — upload an image to Feishu, returning the
  ``image_key`` to be referenced in subsequent messages.

The SDK handles tenant-token acquisition/caching, signing, and retry on its
own; we never touch raw HTTP from here. All functions raise :class:`FeishuError`
when the API returns a failure response.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from typing import Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateImageRequest,
    CreateImageRequestBody,
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageRequest,
    GetMessageResourceRequest,
    ListMessageRequest,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from .log_config import get_logger


log = get_logger(__name__)


class FeishuError(RuntimeError):
    """Raised for any failure talking to the Feishu Open Platform."""


_client_lock = threading.Lock()
_client: Optional[lark.Client] = None


def _build_client() -> lark.Client:
    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")
    if not app_id or not app_secret:
        raise FeishuError(
            "FEISHU_APP_ID and FEISHU_APP_SECRET must be set. Create a self-built "
            "app at https://open.feishu.cn/app and configure them in your .env."
        )
    log_level = lark.LogLevel.INFO
    if os.environ.get("FEISHU_DEBUG"):
        log_level = lark.LogLevel.DEBUG
    return (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .log_level(log_level)
        .build()
    )


def get_client() -> lark.Client:
    """Return the process-wide :class:`lark.Client`, building it on first use."""
    global _client
    with _client_lock:
        if _client is None:
            _client = _build_client()
        return _client


def reset_client() -> None:
    """Test hook: drop the cached client so the next call re-reads env."""
    global _client
    with _client_lock:
        _client = None


def _check(resp, op: str) -> None:
    if not resp.success():
        log_id = ""
        try:
            log_id = resp.get_log_id() or ""
        except Exception:  # noqa: BLE001
            pass
        log.error(
            "feishu.api_error",
            op=op,
            code=resp.code,
            msg=resp.msg,
            log_id=log_id,
        )
        raise FeishuError(
            f"feishu {op} failed: code={resp.code} msg={resp.msg!r} log_id={log_id}"
        )
    log.debug("feishu.api_ok", op=op, code=getattr(resp, "code", None))


def _safe_uuid(*parts: str, max_len: int = 50) -> str:
    """Build a deterministic idempotency key, truncating to Feishu's limit."""
    raw = ":".join(p for p in parts if p)
    if len(raw) <= max_len:
        return raw
    # Preserve a prefix for human readability + a stable suffix for uniqueness.
    import hashlib

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    head_len = max_len - 1 - len(digest)
    return f"{raw[:head_len]}-{digest}"


# ---------------------------------------------------------------------------
# Reply (thread under an existing message)
# ---------------------------------------------------------------------------

def reply_text(message_id: str, text: str, *, uuid: Optional[str] = None) -> None:
    log.debug(
        "feishu.reply_text.start",
        message_id=message_id,
        chars=len(text or ""),
        has_uuid=bool(uuid),
    )
    body_builder = ReplyMessageRequestBody.builder().content(
        json.dumps({"text": text}, ensure_ascii=False)
    ).msg_type("text")
    if uuid:
        body_builder = body_builder.uuid(_safe_uuid(uuid))
    req = (
        ReplyMessageRequest.builder()
        .message_id(message_id)
        .request_body(body_builder.build())
        .build()
    )
    started = time.monotonic()
    resp = get_client().im.v1.message.reply(req)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _check(resp, "message.reply(text)")
    log.info(
        "feishu.reply_text.ok",
        message_id=message_id,
        chars=len(text or ""),
        elapsed_ms=elapsed_ms,
    )


def reply_image(message_id: str, image_bytes: bytes, *, uuid: Optional[str] = None) -> str:
    log.debug(
        "feishu.reply_image.start",
        message_id=message_id,
        size_bytes=len(image_bytes) if image_bytes else 0,
        has_uuid=bool(uuid),
    )
    image_key = upload_image(image_bytes)
    body_builder = ReplyMessageRequestBody.builder().content(
        json.dumps({"image_key": image_key}, ensure_ascii=False)
    ).msg_type("image")
    if uuid:
        body_builder = body_builder.uuid(_safe_uuid(uuid))
    req = (
        ReplyMessageRequest.builder()
        .message_id(message_id)
        .request_body(body_builder.build())
        .build()
    )
    started = time.monotonic()
    resp = get_client().im.v1.message.reply(req)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _check(resp, "message.reply(image)")
    log.info(
        "feishu.reply_image.ok",
        message_id=message_id,
        image_key=image_key,
        size_bytes=len(image_bytes),
        elapsed_ms=elapsed_ms,
    )
    return image_key


# ---------------------------------------------------------------------------
# Send (create a new message addressed by chat_id / open_id / etc.)
# ---------------------------------------------------------------------------

def send_text(receive_id: str, text: str, *, receive_id_type: str = "chat_id",
              uuid: Optional[str] = None) -> None:
    log.debug(
        "feishu.send_text.start",
        receive_id=receive_id,
        receive_id_type=receive_id_type,
        chars=len(text or ""),
        has_uuid=bool(uuid),
    )
    body_builder = (
        CreateMessageRequestBody.builder()
        .receive_id(receive_id)
        .msg_type("text")
        .content(json.dumps({"text": text}, ensure_ascii=False))
    )
    if uuid:
        body_builder = body_builder.uuid(_safe_uuid(uuid))
    req = (
        CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(body_builder.build())
        .build()
    )
    started = time.monotonic()
    resp = get_client().im.v1.message.create(req)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _check(resp, "message.create(text)")
    log.info(
        "feishu.send_text.ok",
        receive_id=receive_id,
        chars=len(text or ""),
        elapsed_ms=elapsed_ms,
    )


def send_image(receive_id: str, image_bytes: bytes, *, receive_id_type: str = "chat_id",
               uuid: Optional[str] = None) -> str:
    log.debug(
        "feishu.send_image.start",
        receive_id=receive_id,
        receive_id_type=receive_id_type,
        size_bytes=len(image_bytes) if image_bytes else 0,
        has_uuid=bool(uuid),
    )
    image_key = upload_image(image_bytes)
    body_builder = (
        CreateMessageRequestBody.builder()
        .receive_id(receive_id)
        .msg_type("image")
        .content(json.dumps({"image_key": image_key}, ensure_ascii=False))
    )
    if uuid:
        body_builder = body_builder.uuid(_safe_uuid(uuid))
    req = (
        CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(body_builder.build())
        .build()
    )
    started = time.monotonic()
    resp = get_client().im.v1.message.create(req)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _check(resp, "message.create(image)")
    log.info(
        "feishu.send_image.ok",
        receive_id=receive_id,
        image_key=image_key,
        size_bytes=len(image_bytes),
        elapsed_ms=elapsed_ms,
    )
    return image_key


# ---------------------------------------------------------------------------
# Resources (download attachments, upload images)
# ---------------------------------------------------------------------------

def download_message_resource(message_id: str, file_key: str, *, type_: str = "image") -> bytes:
    """Download an image (or file) attached to a received message."""
    log.debug(
        "feishu.download_resource.start",
        message_id=message_id,
        file_key=file_key,
        type=type_,
    )
    req = (
        GetMessageResourceRequest.builder()
        .message_id(message_id)
        .file_key(file_key)
        .type(type_)
        .build()
    )
    started = time.monotonic()
    resp = get_client().im.v1.message_resource.get(req)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _check(resp, "message_resource.get")
    if resp.file is None:
        log.error(
            "feishu.download_resource.no_file",
            message_id=message_id,
            file_key=file_key,
        )
        raise FeishuError(
            f"feishu message_resource.get returned no file (message_id={message_id}, "
            f"file_key={file_key})"
        )
    try:
        payload = resp.file.read()
        log.info(
            "feishu.download_resource.ok",
            message_id=message_id,
            file_key=file_key,
            type=type_,
            size_bytes=len(payload),
            elapsed_ms=elapsed_ms,
        )
        return payload
    finally:
        try:
            resp.file.close()
        except Exception:  # noqa: BLE001
            pass


def list_chat_messages(
    chat_id: str,
    *,
    count: int = 20,
    start_time_seconds: Optional[int] = None,
    sort_desc: bool = True,
) -> list:
    """Fetch the most recent messages in ``chat_id`` (newest first by default).

    Wraps ``im.v1.message.list``. ``start_time_seconds`` filters by message
    create time (Unix seconds). When ``sort_desc=True`` (the default) the
    Feishu API returns newest-first; the returned list is then reversed
    in-place so callers receive an *oldest → newest* sequence, which is
    convenient for building chat-completion conversation history.

    Used by the bot to surface conversation context when the user @-mentions
    the bot (group) or DMs it (p2p). Required scope: ``im:message:readonly``
    (or ``im:message``).

    Raises :class:`FeishuError` on API failure. Callers should catch and
    degrade gracefully — a context-load failure must never break the
    primary reply path.
    """
    if not chat_id:
        raise FeishuError("list_chat_messages: chat_id is required")
    page_size = max(1, min(int(count), 50))  # Feishu caps page_size at 50.
    log.debug(
        "feishu.list_messages.start",
        chat_id=chat_id,
        page_size=page_size,
        start_time_seconds=start_time_seconds,
        sort_desc=sort_desc,
    )
    builder = (
        ListMessageRequest.builder()
        .container_id_type("chat")
        .container_id(chat_id)
        .page_size(page_size)
        .sort_type("ByCreateTimeDesc" if sort_desc else "ByCreateTimeAsc")
    )
    if start_time_seconds is not None and start_time_seconds > 0:
        builder = builder.start_time(str(int(start_time_seconds)))
    req = builder.build()
    started = time.monotonic()
    resp = get_client().im.v1.message.list(req)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _check(resp, "message.list")
    items = getattr(resp.data, "items", None) if resp.data is not None else None
    items = list(items or [])
    if sort_desc:
        items.reverse()  # oldest → newest for downstream conversation building.
    log.info(
        "feishu.list_messages.ok",
        chat_id=chat_id,
        count=len(items),
        elapsed_ms=elapsed_ms,
    )
    return items


def get_message(message_id: str):
    """Fetch a message by id; returns the first item in the response.

    Used to read the *quoted* message referenced by ``EventMessage.parent_id``
    so we can extract its image_key(s) and feed the picture into ``it2i``.

    Raises :class:`FeishuError` on API failure or empty result. Callers should
    catch it and degrade gracefully (the user can still get a text reply).
    """
    log.debug("feishu.get_message.start", message_id=message_id)
    req = (
        GetMessageRequest.builder()
        .message_id(message_id)
        .build()
    )
    started = time.monotonic()
    resp = get_client().im.v1.message.get(req)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _check(resp, "message.get")
    items = getattr(resp.data, "items", None) if resp.data is not None else None
    if not items:
        log.error(
            "feishu.get_message.empty",
            message_id=message_id,
            elapsed_ms=elapsed_ms,
        )
        raise FeishuError(
            f"feishu message.get returned no items (message_id={message_id})"
        )
    log.info(
        "feishu.get_message.ok",
        message_id=message_id,
        elapsed_ms=elapsed_ms,
    )
    return items[0]


def _detect_image_ext(image_bytes: bytes) -> str:
    """Sniff a small set of common image formats. Defaults to ``jpg``."""
    head = bytes(image_bytes[:12])
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return "jpg"


def upload_image(image_bytes: bytes, *, file_name: Optional[str] = None) -> str:
    """Upload bytes as an image; return the resulting ``image_key``.

    Feishu's image upload is multipart/form-data and the SDK's serializer only
    accepts the file part when it's an ``io.IOBase`` *with* a ``.name`` set;
    a bare ``BytesIO`` is silently dropped and the server replies with
    ``234001 / Invalid request param``. We always wrap the bytes and set a
    sensible filename so the multipart part is well-formed.
    """
    if not image_bytes:
        raise FeishuError("upload_image: image_bytes is empty")
    name = file_name or f"image.{_detect_image_ext(image_bytes)}"
    log.debug(
        "feishu.upload_image.start",
        size_bytes=len(image_bytes),
        file_name=name,
    )
    bio = io.BytesIO(image_bytes)
    bio.name = name
    body = (
        CreateImageRequestBody.builder()
        .image_type("message")
        .image(bio)
        .build()
    )
    req = CreateImageRequest.builder().request_body(body).build()
    started = time.monotonic()
    resp = get_client().im.v1.image.create(req)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _check(resp, "image.create")
    image_key = getattr(resp.data, "image_key", None) if resp.data is not None else None
    if not image_key:
        log.error(
            "feishu.upload_image.empty_key",
            file_name=name,
            elapsed_ms=elapsed_ms,
        )
        raise FeishuError("feishu image.create returned empty image_key")
    log.info(
        "feishu.upload_image.ok",
        file_name=name,
        size_bytes=len(image_bytes),
        image_key=image_key,
        elapsed_ms=elapsed_ms,
    )
    return image_key
