"""Feishu bot: long-connection (WebSocket) event subscription.

This module wires up the official ``lark-oapi`` SDK so that the bot:

1. Connects out to Feishu over a WebSocket (no public URL / HTTPS / ngrok needed).
2. Receives ``im.message.receive_v1`` events as ``P2ImMessageReceiveV1`` objects.
3. Dedups by ``event_id`` (LRU) to defend against reconnect-redelivery.
4. Offloads slow LLM/image work to a bounded thread pool so the SDK's asyncio
   receive loop stays responsive.
5. Optionally loads recent chat history (``im.v1.message.list``) and passes
   it to the LLM as conversation context. Only triggered in P2P chats and
   when @-mentioned in groups. Historical image attachments are included
   only when ``T2T_MULTIMODAL=1``; otherwise a WARNING is logged.
6. Replies via the IM API through helpers in :mod:`app.feishu`.

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

from . import _messages, agent, feishu, history as history_mod


DEFAULT_RESPOND_MODE = "all"  # Feishu only delivers @-mentioned msgs in groups by default.

# History-context defaults (overridable via env). Kept here so test code
# and ``_build_bot_from_env`` agree.
DEFAULT_HISTORY_COUNT = 20
DEFAULT_HISTORY_WINDOW_MINUTES = 60
DEFAULT_HISTORY_MAX_IMAGES = 3


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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

# Re-exports kept for backwards compatibility with existing tests and
# downstream callers. The implementations live in :mod:`app._messages`
# to avoid an ``app.bot`` ⇄ ``app.history`` import cycle.
MENTION_PLACEHOLDER_PREFIX = _messages.MENTION_PLACEHOLDER_PREFIX
_parse_message_content = _messages.parse_message_content
_parse_content = _messages.parse_event_message
_parse_quoted_message = _messages.parse_fetched_message
_strip_mentions = _messages.strip_mentions


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
        history_enabled: bool = True,
        history_count: int = DEFAULT_HISTORY_COUNT,
        history_window_minutes: int = DEFAULT_HISTORY_WINDOW_MINUTES,
        history_max_images: int = DEFAULT_HISTORY_MAX_IMAGES,
        t2t_multimodal: bool = False,
    ) -> None:
        self.respond_mode = respond_mode
        self.bot_open_id = bot_open_id
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, worker_threads),
            thread_name_prefix="feishu-bot",
        )
        self._dedup = LRUSet(capacity=dedup_capacity)
        self._log_level = log_level or lark.LogLevel.INFO
        # History-context settings (see README + AGENTS.md for env vars).
        self.history_enabled = history_enabled
        self.history_count = max(1, int(history_count))
        self.history_window_minutes = max(0, int(history_window_minutes))
        self.history_max_images = max(0, int(history_max_images))
        self.t2t_multimodal = t2t_multimodal

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

        # Capture extra context for history loading (sender, chat type,
        # whether the bot was @-mentioned). Done here on the asyncio loop
        # since the live ``EventMessage`` is cheap to inspect.
        chat_type = getattr(message, "chat_type", None)
        sender_open_id = self._extract_sender_open_id(sender)
        is_mentioned = self._was_bot_mentioned(message)

        # Hand off to a worker thread so the asyncio loop stays free.
        self._executor.submit(
            self._process,
            event_id=event_id,
            message_id=message.message_id,
            chat_id=message.chat_id,
            text=text,
            image_keys=image_keys,
            parent_id=parent_id,
            chat_type=chat_type,
            sender_open_id=sender_open_id,
            is_mentioned=is_mentioned,
        )

    # -- mention gating -----------------------------------------------------

    @staticmethod
    def _extract_sender_open_id(sender) -> Optional[str]:
        if sender is None:
            return None
        sid = getattr(sender, "sender_id", None)
        return getattr(sid, "open_id", None) if sid is not None else None

    def _was_bot_mentioned(self, message) -> bool:
        if not self.bot_open_id:
            return False
        for m in (message.mentions or []):
            mid = getattr(m, "id", None)
            open_id = getattr(mid, "open_id", None) if mid else None
            if open_id == self.bot_open_id:
                return True
        return False

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
                 parent_id: Optional[str] = None,
                 chat_type: Optional[str] = None,
                 sender_open_id: Optional[str] = None,
                 is_mentioned: bool = False) -> None:
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

            # 3) Optional: load recent chat history as conversation context.
            history_result = self._maybe_load_history(
                chat_id=chat_id,
                current_message_id=message_id,
                chat_type=chat_type,
                sender_open_id=sender_open_id,
                is_mentioned=is_mentioned,
            )

            agent.handle_feishu_event(
                text=text,
                image_bytes=image_bytes,
                message_id=message_id,
                chat_id=chat_id,
                event_id=event_id,
                history=history_result,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[bot] worker error processing {event_id}: {e}", file=sys.stderr)
            self._safe_error_reply(message_id, event_id, e)

    # -- history loading ----------------------------------------------------

    def _should_load_history(self, *, chat_type: Optional[str],
                             is_mentioned: bool) -> bool:
        """User-confirmed scope: ``mentions_and_p2p``.

        P2P chats always qualify (the user is talking directly to the bot).
        In groups, history is only loaded when the bot is @-mentioned —
        we don't want random group chatter polluting context. Detecting a
        mention requires ``FEISHU_BOT_OPEN_ID``; if it's unset we
        conservatively skip history in groups (logged at INFO).
        """
        if not self.history_enabled:
            return False
        if chat_type == "p2p":
            return True
        if chat_type and chat_type != "p2p":
            if not self.bot_open_id:
                print(
                    "[history] skipping in group: FEISHU_BOT_OPEN_ID unset, "
                    "cannot reliably detect bot mention",
                    file=sys.stderr,
                )
                return False
            return is_mentioned
        # Unknown chat_type — be conservative.
        return False

    def _maybe_load_history(
        self,
        *,
        chat_id: str,
        current_message_id: str,
        chat_type: Optional[str],
        sender_open_id: Optional[str],
        is_mentioned: bool,
    ) -> Optional["history_mod.HistoryResult"]:
        """Fetch + assemble conversation context. Returns ``None`` on miss.

        All failures degrade silently to "no history" — the primary reply
        path must never break because history loading failed.
        """
        if not self._should_load_history(
            chat_type=chat_type, is_mentioned=is_mentioned
        ):
            return None

        # Time window (Unix seconds). 0 minutes ⇒ unlimited.
        start_time_seconds: Optional[int] = None
        if self.history_window_minutes > 0:
            import time as _time

            start_time_seconds = int(_time.time()) - self.history_window_minutes * 60

        try:
            raw_messages = feishu.list_chat_messages(
                chat_id,
                count=self.history_count,
                start_time_seconds=start_time_seconds,
            )
        except feishu.FeishuError as e:
            print(
                f"[history] list_chat_messages failed (chat_id={chat_id}): {e}; "
                "continuing without history",
                file=sys.stderr,
            )
            return None
        except Exception as e:  # noqa: BLE001
            print(
                f"[history] unexpected error loading history: {e}; "
                "continuing without history",
                file=sys.stderr,
            )
            return None

        if not raw_messages:
            return None

        # Identify which messages carry images and download the most recent
        # ``history_max_images`` of them. We download newest → oldest so the
        # most-relevant images survive the cap.
        image_bytes_by_key: dict = {}
        if self.t2t_multimodal and self.history_max_images > 0:
            image_bytes_by_key = self._download_history_images(
                raw_messages,
                current_message_id,
                current_user_open_id=sender_open_id,
            )

        result = history_mod.build_history(
            raw_messages,
            current_user_open_id=sender_open_id,
            bot_open_id=self.bot_open_id,
            filter_mode="self_and_bot",
            skip_message_ids={current_message_id} if current_message_id else None,
            include_images=self.t2t_multimodal,
            max_images=self.history_max_images,
            image_bytes_by_key=image_bytes_by_key,
        )

        if result.image_count_skipped_no_multimodal > 0:
            print(
                f"[history] WARNING: T2T_MULTIMODAL is not enabled; skipped "
                f"{result.image_count_skipped_no_multimodal} image(s) from "
                f"conversation history. Set T2T_MULTIMODAL=1 to include "
                f"historical images in LLM context.",
                file=sys.stderr,
            )
        return result

    def _download_history_images(
        self,
        raw_messages: list,
        current_message_id: str,
        *,
        current_user_open_id: Optional[str] = None,
    ) -> dict:
        """Download up to ``history_max_images`` historical images.

        Only considers messages that would pass the ``self_and_bot`` filter
        in :func:`history.build_history` — i.e., messages from the current
        user or the bot itself. This prevents third-party users' images in
        a group chat from consuming the image budget (and from being fed
        to the LLM later).

        Walks ``raw_messages`` newest-first (the input is oldest → newest,
        so we iterate ``reversed``) and stops once the cap is reached.
        Per-image failures are logged and skipped — they must not break
        the primary reply path.
        """
        downloaded: dict = {}
        budget = self.history_max_images
        for msg in reversed(raw_messages):
            if budget <= 0:
                break
            msg_id = getattr(msg, "message_id", None)
            if not msg_id or msg_id == current_message_id:
                continue
            if not history_mod._passes_filter(
                msg,
                history_mod._extract_sender_open_id(msg),
                current_user_open_id=current_user_open_id,
                bot_open_id=self.bot_open_id,
                filter_mode="self_and_bot",
            ):
                continue
            _, keys = _parse_quoted_message(msg)
            for key in keys:
                if budget <= 0:
                    break
                if key in downloaded:
                    continue
                try:
                    downloaded[key] = feishu.download_message_resource(
                        msg_id, key, type_="image"
                    )
                    budget -= 1
                except feishu.FeishuError as e:
                    print(
                        f"[history] failed to download history image "
                        f"(msg={msg_id}, key={key}): {e}; skipping",
                        file=sys.stderr,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[history] unexpected error downloading history image: "
                        f"{e}; skipping",
                        file=sys.stderr,
                    )
        return downloaded

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
    workers = _env_int("FEISHU_WORKER_THREADS", 4)
    dedup = _env_int("FEISHU_DEDUP_CAPACITY", 1024)
    log_level = lark.LogLevel.DEBUG if os.environ.get("FEISHU_DEBUG") else lark.LogLevel.INFO
    history_enabled = _env_bool("FEISHU_HISTORY_ENABLED", True)
    history_count = _env_int("FEISHU_HISTORY_COUNT", DEFAULT_HISTORY_COUNT)
    history_window_minutes = _env_int(
        "FEISHU_HISTORY_WINDOW_MINUTES", DEFAULT_HISTORY_WINDOW_MINUTES
    )
    history_max_images = _env_int(
        "FEISHU_HISTORY_MAX_IMAGES", DEFAULT_HISTORY_MAX_IMAGES
    )
    t2t_multimodal = _env_bool("T2T_MULTIMODAL", False)
    return Bot(
        respond_mode=respond_mode,
        bot_open_id=bot_open_id,
        worker_threads=workers,
        dedup_capacity=dedup,
        log_level=log_level,
        history_enabled=history_enabled,
        history_count=history_count,
        history_window_minutes=history_window_minutes,
        history_max_images=history_max_images,
        t2t_multimodal=t2t_multimodal,
    )


def start() -> None:
    """Build a :class:`Bot` from environment and start it. Blocks."""
    _build_bot_from_env().start()
