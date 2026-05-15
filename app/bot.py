"""Feishu bot: long-connection (WebSocket) event subscription.

This module wires up the official ``lark-oapi`` SDK so that the bot:

1. Connects out to Feishu over a WebSocket (no public URL / HTTPS / ngrok needed).
2. Receives ``im.message.receive_v1`` events as ``P2ImMessageReceiveV1`` objects.
3. Dedups by ``event_id`` (LRU) to defend against reconnect-redelivery.
4. Offloads slow LLM/image work to a bounded thread pool so the SDK's asyncio
   receive loop stays responsive.
5. Replies via the IM API through helpers in :mod:`app.feishu`.

The handler is deliberately small and defensive — on any worker exception it
attempts a best-effort short error reply so the user isn't left hanging.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

from . import agent, feishu


DEFAULT_RESPOND_MODE = "all"  # Feishu only delivers @-mentioned msgs in groups by default.


# ---------------------------------------------------------------------------
# LRU dedup set
# ---------------------------------------------------------------------------

class LRUSet:
    """Tiny FIFO set used to dedupe Feishu ``event_id`` values."""

    def __init__(self, capacity: int = 1024) -> None:
        self._capacity = max(1, capacity)
        self._items: dict = {}
        self._lock = threading.Lock()

    def add(self, key: str) -> bool:
        """Return True if ``key`` was newly added, False if it was already present."""
        with self._lock:
            if key in self._items:
                # Refresh recency.
                self._items.pop(key, None)
                self._items[key] = True
                return False
            self._items[key] = True
            while len(self._items) > self._capacity:
                self._items.pop(next(iter(self._items)))
            return True


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------

MENTION_PLACEHOLDER_PREFIX = "@_user_"


def _parse_message_content(msg_type: Optional[str], content: Optional[str]) -> tuple[str, list[str]]:
    """Parse a Feishu message ``(message_type, content)`` pair.

    Supports ``text``, ``image``, and ``post`` (rich text with embedded
    images). Returns empty strings/lists for unsupported types or unparseable
    content. The same shape is produced by ``EventMessage`` (live event) and
    by ``Message.body`` (when we fetch a *quoted* message via
    ``client.im.v1.message.get``), so this helper is used in both code paths.
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


def _parse_content(message) -> tuple[str, list[str]]:
    """Convenience wrapper for live ``EventMessage`` objects."""
    return _parse_message_content(message.message_type, message.content)


def _parse_quoted_message(msg) -> tuple[str, list[str]]:
    """Convenience wrapper for ``Message`` objects returned by ``message.get``.

    The shape differs slightly from the live event: the JSON content is
    nested under ``msg.body.content`` and the type field is ``msg_type``
    rather than ``message_type``.
    """
    body = getattr(msg, "body", None)
    content = getattr(body, "content", None) if body is not None else None
    return _parse_message_content(getattr(msg, "msg_type", None), content)


def _strip_mentions(text: str, mentions) -> str:
    """Replace ``@_user_N`` placeholders so they don't pollute the prompt."""
    if not text or not mentions:
        return text
    cleaned = text
    for m in mentions:
        key = getattr(m, "key", None)
        if key and key.startswith(MENTION_PLACEHOLDER_PREFIX):
            cleaned = cleaned.replace(key, "")
    # Tidy up consecutive whitespace.
    return " ".join(cleaned.split()).strip()


# ---------------------------------------------------------------------------
# Bot wiring
# ---------------------------------------------------------------------------

class Bot:
    """A configured Feishu bot. Use :meth:`start` to begin receiving events."""

    def __init__(
        self,
        *,
        respond_mode: str = DEFAULT_RESPOND_MODE,
        bot_open_id: Optional[str] = None,
        worker_threads: int = 4,
        dedup_capacity: int = 1024,
        log_level: Optional[lark.LogLevel] = None,
    ) -> None:
        self.respond_mode = respond_mode
        self.bot_open_id = bot_open_id
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, worker_threads),
            thread_name_prefix="feishu-bot",
        )
        self._dedup = LRUSet(capacity=dedup_capacity)
        self._log_level = log_level or lark.LogLevel.INFO

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Block forever, receiving Feishu events over WebSocket."""
        app_id = os.environ.get("FEISHU_APP_ID")
        app_secret = os.environ.get("FEISHU_APP_SECRET")
        if not app_id or not app_secret:
            raise SystemExit(
                "FEISHU_APP_ID and FEISHU_APP_SECRET are required to start the bot. "
                "Create a self-built app at https://open.feishu.cn/app and configure "
                "them in your .env."
            )

        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message)
            .build()
        )
        ws_client = lark.ws.Client(
            app_id,
            app_secret,
            event_handler=handler,
            log_level=self._log_level,
        )
        print(
            f"[bot] connecting to Feishu (respond_mode={self.respond_mode}, "
            f"workers={self._executor._max_workers})",
            file=sys.stderr,
        )
        # Blocks until the long connection is torn down.
        ws_client.start()

    # -- SDK callback (runs on the asyncio receive loop) --------------------

    def _on_message(self, data: P2ImMessageReceiveV1) -> None:
        try:
            self._dispatch(data)
        except Exception as e:  # noqa: BLE001
            # Never propagate into the SDK loop.
            print(f"[bot] dispatch error: {e}", file=sys.stderr)

    def _dispatch(self, data: P2ImMessageReceiveV1) -> None:
        event_id = getattr(data.header, "event_id", None) if data.header else None
        sender = data.event.sender
        message = data.event.message

        # Filter out the bot's own messages / system messages.
        sender_type = getattr(sender, "sender_type", None)
        if sender_type and sender_type != "user":
            return

        # Dedup. The SDK may redeliver on reconnect; replies are idempotent at
        # the Feishu API layer via the message uuid, but doing the LLM work
        # twice is wasteful and user-visible.
        if event_id and not self._dedup.add(event_id):
            print(f"[bot] dedup hit, skipping event_id={event_id}", file=sys.stderr)
            return

        text, image_keys = _parse_content(message)
        text = _strip_mentions(text, message.mentions or [])

        # `parent_id` is set when the user *quoted* (引用回复) another message.
        # We only resolve it in the worker thread (it costs a Feishu API call)
        # and only when the current message lacks an inline image.
        parent_id = getattr(message, "parent_id", None) or None

        if not text and not image_keys and not parent_id:
            print(
                f"[bot] message_type={message.message_type} has no text/images and "
                f"no quoted parent; ignoring",
                file=sys.stderr,
            )
            return

        if not self._should_respond(message):
            return

        # Hand off to a worker thread so the asyncio loop stays free.
        self._executor.submit(
            self._process,
            event_id=event_id,
            message_id=message.message_id,
            chat_id=message.chat_id,
            text=text,
            image_keys=image_keys,
            parent_id=parent_id,
        )

    # -- mention gating -----------------------------------------------------

    def _should_respond(self, message) -> bool:
        if self.respond_mode == "all":
            return True
        # mentions_or_p2p: always reply in P2P; in groups, only when mentioned.
        if message.chat_type == "p2p":
            return True
        if not self.bot_open_id:
            # Without a known bot id we can't distinguish a bot mention; be
            # permissive but log so the operator can supply FEISHU_BOT_OPEN_ID.
            print(
                "[bot] respond_mode=mentions_or_p2p but FEISHU_BOT_OPEN_ID is unset; "
                "replying anyway",
                file=sys.stderr,
            )
            return True
        for m in (message.mentions or []):
            mid = getattr(m, "id", None)
            open_id = getattr(mid, "open_id", None) if mid else None
            if open_id == self.bot_open_id:
                return True
        return False

    # -- worker -------------------------------------------------------------

    def _process(self, *, event_id: Optional[str], message_id: str, chat_id: str,
                 text: str, image_keys: list[str],
                 parent_id: Optional[str] = None) -> None:
        image_bytes: Optional[bytes] = None
        image_source_id = message_id
        quoted_image_failure: Optional[str] = None
        try:
            # 1) If no inline image but the user quoted another message, try to
            #    fetch the parent and use *its* image so "改成黑白" + quoted pic
            #    triggers it2i.
            if not image_keys and parent_id:
                image_keys, image_source_id, quoted_image_failure = (
                    self._resolve_quoted_image(parent_id)
                )

            # 2) Download the image bytes (if any).
            if image_keys:
                try:
                    image_bytes = feishu.download_message_resource(
                        image_source_id, image_keys[0], type_="image"
                    )
                except feishu.FeishuError as e:
                    # Distinguish download failure from parent-lookup failure
                    # — we successfully *found* the picture, just can't read
                    # it, so tell the user instead of silently degrading.
                    print(
                        f"[bot] failed to download quoted image "
                        f"(source={image_source_id}, key={image_keys[0]}): {e}",
                        file=sys.stderr,
                    )
                    if image_source_id != message_id:
                        feishu.reply_text(
                            message_id,
                            "⚠️ 无法读取被引用的图片（可能已过期或权限不足），"
                            "请直接重新发一次图片再描述要怎么改。",
                            uuid=f"{event_id or message_id}:quoted-image-error",
                        )
                        return
                    raise

            if quoted_image_failure and not image_bytes:
                # Parent lookup failed and no other image available — fall
                # through to text-only handling but make sure we have *some*
                # prompt to work with.
                print(
                    f"[bot] quoted parent lookup failed: {quoted_image_failure}",
                    file=sys.stderr,
                )

            if not text and image_bytes is None:
                # Nothing actionable — skip silently rather than spamming.
                return

            agent.handle_feishu_event(
                text=text,
                image_bytes=image_bytes,
                message_id=message_id,
                chat_id=chat_id,
                event_id=event_id,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[bot] worker error processing {event_id}: {e}", file=sys.stderr)
            self._safe_error_reply(message_id, event_id, e)

    def _resolve_quoted_image(self, parent_id: str
                              ) -> tuple[list[str], str, Optional[str]]:
        """Look up the quoted (parent) message and pull its image_key(s).

        Returns ``(image_keys, source_message_id, error_message_or_None)``.
        On any failure the keys are empty and the error string is set so the
        caller can decide whether to surface it.
        """
        try:
            parent_msg = feishu.get_message(parent_id)
        except feishu.FeishuError as e:
            return [], parent_id, f"message.get failed for {parent_id}: {e}"
        _, parent_image_keys = _parse_quoted_message(parent_msg)
        if not parent_image_keys:
            return [], parent_id, f"quoted message {parent_id} has no image"
        print(
            f"[bot] using image from quoted message {parent_id} "
            f"(key={parent_image_keys[0]})",
            file=sys.stderr,
        )
        return parent_image_keys, parent_id, None

    def _safe_error_reply(self, message_id: str, event_id: Optional[str], err: Exception) -> None:
        try:
            short = str(err)
            if len(short) > 200:
                short = short[:200] + "…"
            feishu.reply_text(
                message_id,
                f"⚠️ 抱歉，处理消息时出错: {short}",
                uuid=f"{event_id or message_id}:error",
            )
        except Exception as nested:  # noqa: BLE001
            print(f"[bot] failed to deliver error reply: {nested}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _build_bot_from_env() -> Bot:
    respond_mode = os.environ.get("FEISHU_RESPOND_MODE", DEFAULT_RESPOND_MODE).strip() or DEFAULT_RESPOND_MODE
    if respond_mode not in ("all", "mentions_or_p2p"):
        print(
            f"[bot] invalid FEISHU_RESPOND_MODE={respond_mode!r}, falling back to 'all'",
            file=sys.stderr,
        )
        respond_mode = "all"
    bot_open_id = os.environ.get("FEISHU_BOT_OPEN_ID") or None
    workers = int(os.environ.get("FEISHU_WORKER_THREADS", "4"))
    dedup = int(os.environ.get("FEISHU_DEDUP_CAPACITY", "1024"))
    log_level = lark.LogLevel.DEBUG if os.environ.get("FEISHU_DEBUG") else lark.LogLevel.INFO
    return Bot(
        respond_mode=respond_mode,
        bot_open_id=bot_open_id,
        worker_threads=workers,
        dedup_capacity=dedup,
        log_level=log_level,
    )


def start() -> None:
    """Build a :class:`Bot` from environment and start it. Blocks."""
    _build_bot_from_env().start()
