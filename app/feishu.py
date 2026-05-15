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
        raise FeishuError(
            f"feishu {op} failed: code={resp.code} msg={resp.msg!r} log_id={log_id}"
        )


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
    resp = get_client().im.v1.message.reply(req)
    _check(resp, "message.reply(text)")


def reply_image(message_id: str, image_bytes: bytes, *, uuid: Optional[str] = None) -> str:
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
    resp = get_client().im.v1.message.reply(req)
    _check(resp, "message.reply(image)")
    return image_key


# ---------------------------------------------------------------------------
# Send (create a new message addressed by chat_id / open_id / etc.)
# ---------------------------------------------------------------------------

def send_text(receive_id: str, text: str, *, receive_id_type: str = "chat_id",
              uuid: Optional[str] = None) -> None:
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
    resp = get_client().im.v1.message.create(req)
    _check(resp, "message.create(text)")


def send_image(receive_id: str, image_bytes: bytes, *, receive_id_type: str = "chat_id",
               uuid: Optional[str] = None) -> str:
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
    resp = get_client().im.v1.message.create(req)
    _check(resp, "message.create(image)")
    return image_key


# ---------------------------------------------------------------------------
# Resources (download attachments, upload images)
# ---------------------------------------------------------------------------

def download_message_resource(message_id: str, file_key: str, *, type_: str = "image") -> bytes:
    """Download an image (or file) attached to a received message."""
    req = (
        GetMessageResourceRequest.builder()
        .message_id(message_id)
        .file_key(file_key)
        .type(type_)
        .build()
    )
    resp = get_client().im.v1.message_resource.get(req)
    _check(resp, "message_resource.get")
    if resp.file is None:
        raise FeishuError(
            f"feishu message_resource.get returned no file (message_id={message_id}, "
            f"file_key={file_key})"
        )
    try:
        return resp.file.read()
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
    resp = get_client().im.v1.message.list(req)
    _check(resp, "message.list")
    items = getattr(resp.data, "items", None) if resp.data is not None else None
    items = list(items or [])
    if sort_desc:
        items.reverse()  # oldest → newest for downstream conversation building.
    return items


def get_message(message_id: str):
    """Fetch a message by id; returns the first item in the response.

    Used to read the *quoted* message referenced by ``EventMessage.parent_id``
    so we can extract its image_key(s) and feed the picture into ``it2i``.

    Raises :class:`FeishuError` on API failure or empty result. Callers should
    catch it and degrade gracefully (the user can still get a text reply).
    """
    req = (
        GetMessageRequest.builder()
        .message_id(message_id)
        .build()
    )
    resp = get_client().im.v1.message.get(req)
    _check(resp, "message.get")
    items = getattr(resp.data, "items", None) if resp.data is not None else None
    if not items:
        raise FeishuError(
            f"feishu message.get returned no items (message_id={message_id})"
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
    bio = io.BytesIO(image_bytes)
    bio.name = name
    body = (
        CreateImageRequestBody.builder()
        .image_type("message")
        .image(bio)
        .build()
    )
    req = CreateImageRequest.builder().request_body(body).build()
    resp = get_client().im.v1.image.create(req)
    _check(resp, "image.create")
    image_key = getattr(resp.data, "image_key", None) if resp.data is not None else None
    if not image_key:
        raise FeishuError("feishu image.create returned empty image_key")
    return image_key
