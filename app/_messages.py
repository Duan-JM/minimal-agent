"""Shared Feishu message-content parsers.

Lives outside :mod:`app.bot` so that downstream helpers (notably
:mod:`app.history`) can reuse the same parsing logic without creating an
``app.bot`` ⇄ ``app.history`` import cycle. The parsers handle the three
content shapes the bot cares about — ``text``, ``image``, and ``post`` —
and work for both live ``EventMessage`` objects (``message_type`` +
``content``) and ``Message`` objects fetched via ``im.v1.message.get`` or
``im.v1.message.list`` (``msg_type`` + ``body.content``).
"""
from __future__ import annotations

import json
from typing import Optional


MENTION_PLACEHOLDER_PREFIX = "@_user_"


def parse_message_content(
    msg_type: Optional[str], content: Optional[str]
) -> tuple[str, list[str]]:
    """Parse a Feishu message ``(message_type, content)`` pair.

    Returns ``(text, image_keys)``. Empty values for unsupported types or
    unparseable content. Supports ``text``, ``image``, and ``post`` (rich
    text with embedded images).
    """
    if not content:
        return "", []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return "", []

    if msg_type == "text":
        return (data.get("text") or "").strip(), []

    if msg_type == "image":
        key = data.get("image_key") or ""
        return "", [key] if key else []

    if msg_type == "post":
        text_parts: list[str] = []
        image_keys: list[str] = []
        nodes = data.get("content") or []
        for line in nodes:
            for node in line or []:
                tag = node.get("tag")
                if tag == "text":
                    t = node.get("text") or ""
                    if t:
                        text_parts.append(t)
                elif tag == "img" or tag == "image":
                    k = node.get("image_key") or ""
                    if k:
                        image_keys.append(k)
        return " ".join(text_parts).strip(), image_keys

    return "", []


def parse_event_message(message) -> tuple[str, list[str]]:
    """Parse a live ``EventMessage`` (``message_type`` + ``content``)."""
    return parse_message_content(
        getattr(message, "message_type", None),
        getattr(message, "content", None),
    )


def parse_fetched_message(msg) -> tuple[str, list[str]]:
    """Parse a ``Message`` returned by ``message.get`` / ``message.list``.

    The shape differs slightly from the live event: the JSON content is
    nested under ``msg.body.content`` and the type field is ``msg_type``
    rather than ``message_type``.
    """
    body = getattr(msg, "body", None)
    content = getattr(body, "content", None) if body is not None else None
    return parse_message_content(getattr(msg, "msg_type", None), content)


def strip_mentions(text: str, mentions) -> str:
    """Replace ``@_user_N`` placeholders so they don't pollute the prompt."""
    if not text or not mentions:
        return text
    cleaned = text
    for m in mentions:
        key = getattr(m, "key", None)
        if key and key.startswith(MENTION_PLACEHOLDER_PREFIX):
            cleaned = cleaned.replace(key, "")
    return " ".join(cleaned.split()).strip()
