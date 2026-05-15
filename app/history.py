"""Build OpenAI chat-completion conversation history from Feishu messages.

The bot calls :func:`build_history` to turn a list of Feishu ``Message``
objects (as returned by ``feishu.list_chat_messages``) into:

- ``chat_messages``: a list of OpenAI ``messages``-style dicts (with
  ``role`` and ``content`` — content is either a plain string or, for
  multimodal turns, a list of ``{"type": "text"|"image_url", ...}``
  parts). This is prepended to the current turn for ``t2t`` / ``it2t``.
- ``text_summary``: a textual rendering used by ``t2i`` / ``t2i`` (image
  generation endpoints can't accept multimodal turns).

Multimodal handling is controlled by the ``include_images`` flag (driven
by ``T2T_MULTIMODAL`` at the call site). When False, images in history
are dropped from ``chat_messages`` and counted in
``image_count_skipped_no_multimodal`` so the caller can log a single
WARNING. Image bytes themselves are downloaded by the bot worker (which
has the message_id needed for ``message_resource.get``) and passed in
via ``image_bytes_by_key``.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Optional

from ._messages import parse_fetched_message


@dataclass
class HistoryResult:
    """The processed conversation context handed to the tools layer."""

    chat_messages: list = field(default_factory=list)
    text_summary: str = ""
    image_count_total: int = 0
    image_count_included: int = 0
    image_count_skipped_no_multimodal: int = 0

    @property
    def has_content(self) -> bool:
        return bool(self.chat_messages) or bool(self.text_summary)


_FILTER_MODES = ("self_and_bot", "all_users", "self_only")


def _detect_image_mime(image_bytes: bytes) -> str:
    head = bytes(image_bytes[:12])
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _data_uri(image_bytes: bytes) -> str:
    mime = _detect_image_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


def _extract_sender_open_id(msg) -> Optional[str]:
    """Pull ``sender.id.open_id`` (with a fallback for live-event shape).

    ``Message.sender`` (list/get API) exposes ``sender.id.open_id``;
    ``EventMessage.sender`` exposes ``sender.sender_id.open_id``. We try
    both so :func:`build_history` works on either shape.
    """
    sender = getattr(msg, "sender", None)
    if sender is None:
        return None
    sid = getattr(sender, "id", None)
    open_id = getattr(sid, "open_id", None) if sid is not None else None
    if open_id:
        return open_id
    sid2 = getattr(sender, "sender_id", None)
    return getattr(sid2, "open_id", None) if sid2 is not None else None


def _sender_type(msg) -> Optional[str]:
    """Return ``sender.sender_type`` if present (``"user"`` vs ``"app"``)."""
    sender = getattr(msg, "sender", None)
    return getattr(sender, "sender_type", None) if sender is not None else None


def is_bot_message(msg, bot_open_id: Optional[str]) -> bool:
    """True when ``msg`` was authored by the bot itself.

    Prefers an open_id match (most reliable, works in groups with multiple
    apps). Falls back to ``sender_type == "app"`` so 1-on-1 chats still
    surface the bot's own replies when ``FEISHU_BOT_OPEN_ID`` is unset.
    """
    sender_open_id = _extract_sender_open_id(msg)
    if bot_open_id and sender_open_id == bot_open_id:
        return True
    if not bot_open_id and _sender_type(msg) == "app":
        return True
    return False


def _role_for(msg, sender_open_id: Optional[str],
              bot_open_id: Optional[str]) -> str:
    if is_bot_message(msg, bot_open_id):
        return "assistant"
    return "user"


def _passes_filter(
    msg,
    sender_open_id: Optional[str],
    *,
    current_user_open_id: Optional[str],
    bot_open_id: Optional[str],
    filter_mode: str,
) -> bool:
    if filter_mode == "all_users":
        return True
    if filter_mode == "self_only":
        return bool(sender_open_id) and sender_open_id == current_user_open_id
    # default: self_and_bot
    if is_bot_message(msg, bot_open_id):
        return True
    if current_user_open_id and sender_open_id == current_user_open_id:
        return True
    return False


def build_history(
    messages: list,
    *,
    current_user_open_id: Optional[str],
    bot_open_id: Optional[str],
    filter_mode: str = "self_and_bot",
    skip_message_ids: Optional[set] = None,
    include_images: bool = False,
    max_images: int = 3,
    image_bytes_by_key: Optional[dict] = None,
) -> HistoryResult:
    """Fold Feishu ``Message`` objects into a :class:`HistoryResult`.

    ``messages`` must be in chronological order (oldest → newest), which
    is what :func:`app.feishu.list_chat_messages` produces.

    - ``current_user_open_id``: the open_id of the user the bot is
      currently replying to. Used for role tagging and filtering.
    - ``bot_open_id``: the bot's own open_id. Messages from this sender
      become ``role="assistant"``.
    - ``filter_mode``: ``"self_and_bot"`` (default — only the user and the
      bot), ``"all_users"`` (everyone in the chat), or ``"self_only"``
      (just the user).
    - ``skip_message_ids``: drop these message ids (typically contains
      the message we're currently replying to so we don't echo it back).
    - ``include_images``: when False, all image attachments in history
      are skipped from ``chat_messages`` and counted in
      ``image_count_skipped_no_multimodal``. The caller is expected to
      emit one WARNING based on that count.
    - ``max_images``: hard cap on how many history images make it into
      ``chat_messages`` (cheaper LLM calls, smaller payloads). Counted
      across all messages combined.
    - ``image_bytes_by_key``: pre-downloaded image bytes keyed by
      Feishu ``image_key``. Images whose key isn't in this dict are
      effectively dropped (and not counted as "included").
    """
    if filter_mode not in _FILTER_MODES:
        filter_mode = "self_and_bot"
    result = HistoryResult()
    skip = skip_message_ids or set()
    image_bytes_by_key = image_bytes_by_key or {}
    summary_lines: list[str] = []
    images_used = 0

    for msg in messages:
        message_id = getattr(msg, "message_id", None)
        if message_id and message_id in skip:
            continue

        sender_open_id = _extract_sender_open_id(msg)
        if not _passes_filter(
            msg,
            sender_open_id,
            current_user_open_id=current_user_open_id,
            bot_open_id=bot_open_id,
            filter_mode=filter_mode,
        ):
            continue

        text, image_keys = parse_fetched_message(msg)
        if not text and not image_keys:
            continue

        role = _role_for(msg, sender_open_id, bot_open_id)
        result.image_count_total += len(image_keys)

        if image_keys and include_images and images_used < max_images:
            # Build multimodal content parts. Drop image_keys we have no
            # bytes for — they're effectively unavailable.
            parts: list = []
            for key in image_keys:
                if images_used >= max_images:
                    break
                blob = image_bytes_by_key.get(key)
                if not blob:
                    continue
                parts.append(
                    {"type": "image_url", "image_url": {"url": _data_uri(blob)}}
                )
                images_used += 1
                result.image_count_included += 1
            if text:
                parts.append({"type": "text", "text": text})
            if parts:
                result.chat_messages.append({"role": role, "content": parts})
        else:
            if image_keys and not include_images:
                result.image_count_skipped_no_multimodal += len(image_keys)
            display_text = text if text else ("[image]" if image_keys else "")
            if display_text:
                result.chat_messages.append({"role": role, "content": display_text})

        # Always update text_summary (used by t2i/it2i, which don't take
        # multimodal turns). Images are represented by <image> placeholders.
        summary_text = text
        if image_keys:
            placeholders = " ".join(["<image>"] * len(image_keys))
            summary_text = (
                f"{summary_text} {placeholders}".strip()
                if summary_text
                else placeholders
            )
        if summary_text:
            summary_lines.append(f"[{role}]: {summary_text}")

    result.text_summary = "\n".join(summary_lines)
    return result
