"""Tests for the long-connection bot (app.bot)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def _make_event(
    *,
    message_type="text",
    content=None,
    chat_type="p2p",
    chat_id="oc_chat",
    message_id="om_msg",
    sender_open_id="ou_user",
    sender_type="user",
    event_id="ev1",
    mentions=None,
    parent_id=None,
):
    content_dict = content if content is not None else {"text": "hello"}
    data = mock.MagicMock()
    data.header.event_id = event_id
    data.event.sender.sender_id.open_id = sender_open_id
    data.event.sender.sender_type = sender_type
    data.event.message.message_id = message_id
    data.event.message.chat_id = chat_id
    data.event.message.chat_type = chat_type
    data.event.message.message_type = message_type
    data.event.message.content = json.dumps(content_dict)
    data.event.message.mentions = mentions or []
    # `parent_id` is None unless the user used Feishu's quote/reply UI.
    # We set it explicitly so MagicMock doesn't conjure a truthy auto-attr.
    data.event.message.parent_id = parent_id
    return data


class LRUSetTests(unittest.TestCase):
    def test_add_returns_true_first_time_false_after(self):
        from app.bot import LRUSet
        s = LRUSet(capacity=3)
        self.assertTrue(s.add("a"))
        self.assertFalse(s.add("a"))

    def test_capacity_evicts_oldest(self):
        from app.bot import LRUSet
        s = LRUSet(capacity=2)
        self.assertTrue(s.add("a"))
        self.assertTrue(s.add("b"))
        self.assertTrue(s.add("c"))  # evicts a
        self.assertTrue(s.add("a"))  # a was evicted; treated as new


class ParseContentTests(unittest.TestCase):
    def test_text_message(self):
        from app.bot import _parse_content
        msg = mock.MagicMock(message_type="text", content=json.dumps({"text": "hi there"}))
        text, keys = _parse_content(msg)
        self.assertEqual(text, "hi there")
        self.assertEqual(keys, [])

    def test_image_message(self):
        from app.bot import _parse_content
        msg = mock.MagicMock(message_type="image", content=json.dumps({"image_key": "img_a"}))
        text, keys = _parse_content(msg)
        self.assertEqual(text, "")
        self.assertEqual(keys, ["img_a"])

    def test_post_message_flattens(self):
        from app.bot import _parse_content
        post_content = {
            "title": "Hi",
            "content": [
                [
                    {"tag": "text", "text": "describe"},
                    {"tag": "img", "image_key": "img_a"},
                    {"tag": "text", "text": "this please"},
                ]
            ],
        }
        msg = mock.MagicMock(message_type="post", content=json.dumps(post_content))
        text, keys = _parse_content(msg)
        self.assertIn("describe", text)
        self.assertIn("this please", text)
        self.assertEqual(keys, ["img_a"])

    def test_unsupported_returns_empty(self):
        from app.bot import _parse_content
        msg = mock.MagicMock(message_type="audio", content=json.dumps({"file_key": "f"}))
        text, keys = _parse_content(msg)
        self.assertEqual(text, "")
        self.assertEqual(keys, [])

    def test_strip_mentions(self):
        from app.bot import _strip_mentions
        m = mock.MagicMock(key="@_user_1")
        self.assertEqual(_strip_mentions("@_user_1 hello world", [m]), "hello world")


class _SyncExecutor:
    """Stand-in for ThreadPoolExecutor that runs callables in the same thread."""

    def __init__(self):
        self._max_workers = 1

    def submit(self, fn, *args, **kwargs):  # pragma: no cover - thin shim
        fn(*args, **kwargs)
        return mock.MagicMock()


class BotDispatchTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = mock.patch.dict(
            os.environ,
            {
                "FEISHU_APP_ID": "id",
                "FEISHU_APP_SECRET": "secret",
                "ENABLE_LLM_ROUTER": "0",
            },
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def _make_bot(self, **kwargs):
        from app.bot import Bot
        bot = Bot(**kwargs)
        bot._executor = _SyncExecutor()  # run handler inline for deterministic tests
        return bot

    def test_text_message_dispatches_to_agent(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(content={"text": "hello"})
        with mock.patch.object(bot_mod.feishu, "download_message_resource") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        dl.assert_not_called()
        h.assert_called_once()
        kw = h.call_args.kwargs
        self.assertEqual(kw["text"], "hello")
        self.assertIsNone(kw["image_bytes"])
        self.assertEqual(kw["message_id"], "om_msg")
        self.assertEqual(kw["chat_id"], "oc_chat")
        self.assertEqual(kw["event_id"], "ev1")

    def test_image_message_downloads_then_dispatches(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(message_type="image", content={"image_key": "img_a"})
        with mock.patch.object(bot_mod.feishu, "download_message_resource",
                               return_value=b"RAW") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        dl.assert_called_once_with("om_msg", "img_a", type_="image")
        h.assert_called_once()
        self.assertEqual(h.call_args.kwargs["image_bytes"], b"RAW")

    def test_self_message_skipped(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(sender_type="bot")
        with mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        h.assert_not_called()

    def test_dedup_skips_second_delivery(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        with mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(_make_event(event_id="dup-1"))
            bot._dispatch(_make_event(event_id="dup-1"))
        self.assertEqual(h.call_count, 1)

    def test_unsupported_message_type_ignored(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(message_type="audio", content={"file_key": "f"})
        with mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        h.assert_not_called()

    def test_mention_gating_skips_unmentioned_group(self):
        from app import bot as bot_mod
        bot = self._make_bot(respond_mode="mentions_or_p2p", bot_open_id="ou_bot")
        evt = _make_event(chat_type="group", content={"text": "hi"})
        with mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        h.assert_not_called()

    def test_mention_gating_replies_when_mentioned(self):
        from app import bot as bot_mod
        bot = self._make_bot(respond_mode="mentions_or_p2p", bot_open_id="ou_bot")
        mention = mock.MagicMock()
        mention.key = "@_user_1"
        mention.id.open_id = "ou_bot"
        evt = _make_event(
            chat_type="group",
            content={"text": "@_user_1 hi"},
            mentions=[mention],
        )
        with mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        h.assert_called_once()
        # The @_user_1 placeholder must have been stripped from the prompt.
        self.assertEqual(h.call_args.kwargs["text"], "hi")

    def test_mention_gating_always_replies_in_p2p(self):
        from app import bot as bot_mod
        bot = self._make_bot(respond_mode="mentions_or_p2p", bot_open_id="ou_bot")
        evt = _make_event(chat_type="p2p", content={"text": "hi"})
        with mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        h.assert_called_once()

    def test_worker_error_triggers_error_reply(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(content={"text": "boom"})

        def explode(**kwargs):
            raise RuntimeError("LLM down")

        with mock.patch.object(bot_mod.agent, "handle_feishu_event", side_effect=explode), \
             mock.patch.object(bot_mod.feishu, "reply_text") as rt:
            bot._dispatch(evt)
        rt.assert_called_once()
        args, kwargs = rt.call_args
        self.assertEqual(args[0], "om_msg")
        self.assertIn("LLM down", args[1])
        self.assertIn("error", kwargs["uuid"])


class BotQuotedImageTests(unittest.TestCase):
    """Verify the "user quotes an image + asks for an edit" path triggers it2i."""

    def setUp(self):
        self.env_patch = mock.patch.dict(
            os.environ,
            {"FEISHU_APP_ID": "id", "FEISHU_APP_SECRET": "secret", "ENABLE_LLM_ROUTER": "0"},
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def _make_bot(self, **kwargs):
        from app.bot import Bot
        bot = Bot(**kwargs)
        bot._executor = _SyncExecutor()
        return bot

    @staticmethod
    def _quoted_image_msg(image_key="img_parent"):
        """Stand-in for an `im/v1/messages` GET response item (type=image)."""
        parent = mock.MagicMock()
        parent.msg_type = "image"
        parent.body.content = json.dumps({"image_key": image_key})
        return parent

    @staticmethod
    def _quoted_post_msg(image_keys=("img_p1",)):
        parent = mock.MagicMock()
        parent.msg_type = "post"
        parent.body.content = json.dumps({
            "content": [
                [
                    {"tag": "text", "text": "hi"},
                    *({"tag": "img", "image_key": k} for k in image_keys),
                ]
            ]
        })
        return parent

    def test_quote_image_with_edit_prompt_pulls_parent_image(self):
        """Quote an image + say "改成黑白" → fetch parent, download its image, hand off."""
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(
            content={"text": "改成黑白"},
            parent_id="om_parent",
        )
        with mock.patch.object(bot_mod.feishu, "get_message",
                               return_value=self._quoted_image_msg("img_p")) as gm, \
             mock.patch.object(bot_mod.feishu, "download_message_resource",
                               return_value=b"RAW_PARENT") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        gm.assert_called_once_with("om_parent")
        # Image must be downloaded from the *parent* message, not the user's
        # quoting message (which only contains text).
        dl.assert_called_once_with("om_parent", "img_p", type_="image")
        h.assert_called_once()
        kw = h.call_args.kwargs
        self.assertEqual(kw["text"], "改成黑白")
        self.assertEqual(kw["image_bytes"], b"RAW_PARENT")
        # The reply still threads under the user's message (not the parent).
        self.assertEqual(kw["message_id"], "om_msg")

    def test_quote_post_message_extracts_embedded_image(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(content={"text": "make it red"}, parent_id="om_parent")
        with mock.patch.object(bot_mod.feishu, "get_message",
                               return_value=self._quoted_post_msg(("img_pp",))), \
             mock.patch.object(bot_mod.feishu, "download_message_resource",
                               return_value=b"PNG") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        dl.assert_called_once_with("om_parent", "img_pp", type_="image")
        self.assertEqual(h.call_args.kwargs["image_bytes"], b"PNG")

    def test_inline_image_wins_over_quoted_image(self):
        """If the user attached AND quoted an image, prefer the attached one."""
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(
            message_type="image",
            content={"image_key": "img_inline"},
            parent_id="om_parent",
        )
        with mock.patch.object(bot_mod.feishu, "get_message") as gm, \
             mock.patch.object(bot_mod.feishu, "download_message_resource",
                               return_value=b"INLINE") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        gm.assert_not_called()
        dl.assert_called_once_with("om_msg", "img_inline", type_="image")
        h.assert_called_once()

    def test_quote_text_only_parent_falls_back_to_text_only(self):
        """Quoted parent has no image — proceed with text-only handling."""
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(content={"text": "改成黑白"}, parent_id="om_parent")
        parent = mock.MagicMock()
        parent.msg_type = "text"
        parent.body.content = json.dumps({"text": "previously"})
        with mock.patch.object(bot_mod.feishu, "get_message", return_value=parent), \
             mock.patch.object(bot_mod.feishu, "download_message_resource") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        dl.assert_not_called()
        h.assert_called_once()
        self.assertIsNone(h.call_args.kwargs["image_bytes"])
        self.assertEqual(h.call_args.kwargs["text"], "改成黑白")

    def test_quote_parent_fetch_failure_does_not_break_reply(self):
        """If `message.get` fails (e.g. missing scope), keep going text-only."""
        from app import bot as bot_mod
        from app.feishu import FeishuError
        bot = self._make_bot()
        evt = _make_event(content={"text": "改成黑白"}, parent_id="om_parent")
        with mock.patch.object(bot_mod.feishu, "get_message",
                               side_effect=FeishuError("code=99991663 no scope")), \
             mock.patch.object(bot_mod.feishu, "download_message_resource") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        dl.assert_not_called()
        h.assert_called_once()
        self.assertIsNone(h.call_args.kwargs["image_bytes"])

    def test_quote_image_download_failure_surfaces_clear_error(self):
        """Parent lookup OK but downloading its picture fails — tell the user."""
        from app import bot as bot_mod
        from app.feishu import FeishuError
        bot = self._make_bot()
        evt = _make_event(content={"text": "改成黑白"}, parent_id="om_parent")
        with mock.patch.object(bot_mod.feishu, "get_message",
                               return_value=self._quoted_image_msg("img_p")), \
             mock.patch.object(bot_mod.feishu, "download_message_resource",
                               side_effect=FeishuError("expired")), \
             mock.patch.object(bot_mod.feishu, "reply_text") as rt, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        h.assert_not_called()
        rt.assert_called_once()
        args, kwargs = rt.call_args
        self.assertEqual(args[0], "om_msg")
        self.assertIn("引用", args[1])
        self.assertIn("quoted-image-error", kwargs["uuid"])

    def test_dispatch_does_not_call_get_message_inline(self):
        """Parent lookup must happen in the worker, not on the asyncio loop."""
        from app import bot as bot_mod
        from app.bot import Bot
        bot = Bot()

        captured = {}

        class _CapturingExecutor:
            def __init__(self):
                self._max_workers = 1

            def submit(self, fn, *args, **kwargs):
                captured["kwargs"] = kwargs
                return mock.MagicMock()

        bot._executor = _CapturingExecutor()
        evt = _make_event(content={"text": "改成黑白"}, parent_id="om_parent")
        with mock.patch.object(bot_mod.feishu, "get_message") as gm:
            bot._dispatch(evt)
        gm.assert_not_called()  # the worker would have called it
        self.assertEqual(captured["kwargs"]["parent_id"], "om_parent")


class BotEnvBuilderTests(unittest.TestCase):
    def test_invalid_respond_mode_falls_back(self):
        from app import bot as bot_mod
        with mock.patch.dict(
            os.environ,
            {
                "FEISHU_APP_ID": "id",
                "FEISHU_APP_SECRET": "secret",
                "FEISHU_RESPOND_MODE": "wat",
            },
            clear=False,
        ):
            b = bot_mod._build_bot_from_env()
        self.assertEqual(b.respond_mode, "all")

    def test_start_without_credentials_raises(self):
        from app import bot as bot_mod
        with mock.patch.dict(
            os.environ,
            {"FEISHU_APP_ID": "", "FEISHU_APP_SECRET": ""},
            clear=False,
        ):
            b = bot_mod._build_bot_from_env()
            with self.assertRaises(SystemExit):
                b.start()


class BotShouldLoadHistoryTests(unittest.TestCase):
    def _bot(self, **kwargs):
        from app.bot import Bot
        return Bot(**kwargs)

    def test_p2p_always_loads(self):
        b = self._bot(bot_open_id="ou_bot")
        self.assertTrue(
            b._should_load_history(chat_type="p2p", is_mentioned=False)
        )

    def test_group_loads_when_mentioned(self):
        b = self._bot(bot_open_id="ou_bot")
        self.assertTrue(
            b._should_load_history(chat_type="group", is_mentioned=True)
        )

    def test_group_skipped_when_not_mentioned(self):
        b = self._bot(bot_open_id="ou_bot")
        self.assertFalse(
            b._should_load_history(chat_type="group", is_mentioned=False)
        )

    def test_group_skipped_when_bot_open_id_missing(self):
        b = self._bot(bot_open_id=None)
        # Even if upstream thinks the bot was mentioned, we can't trust it
        # without FEISHU_BOT_OPEN_ID — be conservative and skip.
        self.assertFalse(
            b._should_load_history(chat_type="group", is_mentioned=True)
        )

    def test_disabled_flag_short_circuits(self):
        b = self._bot(bot_open_id="ou_bot", history_enabled=False)
        self.assertFalse(
            b._should_load_history(chat_type="p2p", is_mentioned=True)
        )

    def test_unknown_chat_type_conservative(self):
        b = self._bot(bot_open_id="ou_bot")
        self.assertFalse(
            b._should_load_history(chat_type=None, is_mentioned=True)
        )


class BotHistoryWorkerTests(unittest.TestCase):
    """Verify the worker (``_process``) loads history when eligible and
    degrades gracefully when the API misbehaves."""

    def setUp(self):
        self.env_patch = mock.patch.dict(
            os.environ,
            {"FEISHU_APP_ID": "id", "FEISHU_APP_SECRET": "secret",
             "ENABLE_LLM_ROUTER": "0"},
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def _make_bot(self, **kwargs):
        from app.bot import Bot
        defaults = {
            "bot_open_id": "ou_bot",
            "history_count": 5,
            "history_window_minutes": 0,
            "history_max_images": 0,
            "t2t_multimodal": False,
        }
        defaults.update(kwargs)
        bot = Bot(**defaults)
        bot._executor = _SyncExecutor()
        return bot

    @staticmethod
    def _history_msg(message_id, msg_type, content, sender_open_id):
        m = mock.MagicMock()
        m.message_id = message_id
        m.msg_type = msg_type
        m.body.content = json.dumps(content)
        m.sender.id.open_id = sender_open_id
        return m

    def test_p2p_text_message_loads_history(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(content={"text": "next question"})
        history_items = [
            self._history_msg("h1", "text", {"text": "earlier"},
                              sender_open_id="ou_user"),
            self._history_msg("h2", "text", {"text": "earlier reply"},
                              sender_open_id="ou_bot"),
        ]
        with mock.patch.object(bot_mod.feishu, "list_chat_messages",
                               return_value=history_items) as lst, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        lst.assert_called_once()
        # Worker passed a HistoryResult to the agent.
        kw = h.call_args.kwargs
        self.assertIn("history", kw)
        self.assertIsNotNone(kw["history"])
        roles = [m["role"] for m in kw["history"].chat_messages]
        self.assertEqual(roles, ["user", "assistant"])

    def test_group_unmentioned_does_not_load_history(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        evt = _make_event(chat_type="group", content={"text": "chatter"})
        with mock.patch.object(bot_mod.feishu, "list_chat_messages") as lst, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        lst.assert_not_called()
        # handle_feishu_event still called (respond_mode=all by default), but
        # with no history.
        self.assertIsNone(h.call_args.kwargs.get("history"))

    def test_group_mention_triggers_history_load(self):
        from app import bot as bot_mod
        bot = self._make_bot()
        mention = mock.MagicMock()
        mention.key = "@_user_1"
        mention.id.open_id = "ou_bot"
        evt = _make_event(
            chat_type="group",
            content={"text": "@_user_1 hi"},
            mentions=[mention],
        )
        with mock.patch.object(bot_mod.feishu, "list_chat_messages",
                               return_value=[]) as lst, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        lst.assert_called_once()
        h.assert_called_once()

    def test_history_disabled_skips_list_call(self):
        from app import bot as bot_mod
        bot = self._make_bot(history_enabled=False)
        evt = _make_event(content={"text": "hi"})
        with mock.patch.object(bot_mod.feishu, "list_chat_messages") as lst, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        lst.assert_not_called()
        self.assertIsNone(h.call_args.kwargs.get("history"))

    def test_list_failure_degrades_to_no_history(self):
        from app import bot as bot_mod
        from app.feishu import FeishuError
        bot = self._make_bot()
        evt = _make_event(content={"text": "ping"})
        with mock.patch.object(bot_mod.feishu, "list_chat_messages",
                               side_effect=FeishuError("permission denied")), \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        h.assert_called_once()
        self.assertIsNone(h.call_args.kwargs.get("history"))

    def test_multimodal_off_with_history_images_warns(self):
        from app import bot as bot_mod
        bot = self._make_bot(t2t_multimodal=False)
        evt = _make_event(content={"text": "what about that pic?"})
        history_items = [
            self._history_msg("h_img", "image", {"image_key": "img_a"},
                              sender_open_id="ou_user"),
        ]
        with mock.patch.object(bot_mod.feishu, "list_chat_messages",
                               return_value=history_items), \
             mock.patch.object(bot_mod.feishu, "download_message_resource") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event"), \
             mock.patch("sys.stderr") as stderr:
            bot._dispatch(evt)
        # Bot must NOT download history images when multimodal is off — it's
        # wasted bandwidth.
        dl.assert_not_called()
        # And it must log a WARNING naming the skipped image.
        printed = "".join(call.args[0] for call in stderr.write.call_args_list
                          if call.args and isinstance(call.args[0], str))
        self.assertIn("WARNING", printed)
        self.assertIn("T2T_MULTIMODAL", printed)

    def test_multimodal_on_downloads_history_images(self):
        from app import bot as bot_mod
        bot = self._make_bot(t2t_multimodal=True, history_max_images=2)
        evt = _make_event(content={"text": "describe earlier pic"})
        history_items = [
            self._history_msg("h_img", "image", {"image_key": "img_a"},
                              sender_open_id="ou_user"),
        ]
        with mock.patch.object(bot_mod.feishu, "list_chat_messages",
                               return_value=history_items), \
             mock.patch.object(bot_mod.feishu, "download_message_resource",
                               return_value=b"\x89PNG\r\n\x1a\nABCD") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        dl.assert_called_once_with("h_img", "img_a", type_="image")
        kw = h.call_args.kwargs
        hist = kw["history"]
        self.assertEqual(hist.image_count_included, 1)
        self.assertEqual(hist.image_count_skipped_no_multimodal, 0)

    def test_history_image_download_failure_skips_image_only(self):
        from app import bot as bot_mod
        from app.feishu import FeishuError
        bot = self._make_bot(t2t_multimodal=True, history_max_images=2)
        evt = _make_event(content={"text": "ok"})
        history_items = [
            self._history_msg("h_img", "image", {"image_key": "img_a"},
                              sender_open_id="ou_user"),
            self._history_msg("h_text", "text", {"text": "earlier reply"},
                              sender_open_id="ou_bot"),
        ]
        with mock.patch.object(bot_mod.feishu, "list_chat_messages",
                               return_value=history_items), \
             mock.patch.object(bot_mod.feishu, "download_message_resource",
                               side_effect=FeishuError("expired")), \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        # The download failure must not blow up the worker — text portion
        # of history still passes through.
        h.assert_called_once()
        hist = h.call_args.kwargs["history"]
        self.assertEqual(hist.image_count_included, 0)
        # The text-only history message survives.
        contents = [m["content"] for m in hist.chat_messages]
        self.assertIn("earlier reply", contents)

    def test_group_third_party_images_not_downloaded(self):
        """In a group chat, ``_download_history_images`` must skip images
        from third parties (not the current user, not the bot). Otherwise
        the image budget gets consumed by irrelevant pictures and they
        would also leak into LLM context."""
        from app import bot as bot_mod
        bot = self._make_bot(t2t_multimodal=True, history_max_images=2,
                             bot_open_id="ou_bot")
        # Group mention triggers history load.
        mention = mock.MagicMock()
        mention.key = "@_user_1"
        mention.id.open_id = "ou_bot"
        evt = _make_event(
            chat_type="group",
            content={"text": "@_user_1 what did I draw earlier?"},
            mentions=[mention],
        )
        # Newest first: a stranger's image (must be skipped), then the
        # current user's image (must be downloaded).
        history_items = [
            self._history_msg("h_self", "image", {"image_key": "img_user"},
                              sender_open_id="ou_user"),
            self._history_msg("h_stranger", "image",
                              {"image_key": "img_other"},
                              sender_open_id="ou_someone_else"),
        ]
        with mock.patch.object(bot_mod.feishu, "list_chat_messages",
                               return_value=history_items), \
             mock.patch.object(bot_mod.feishu, "download_message_resource",
                               return_value=b"\x89PNG\r\n\x1a\nPIC") as dl, \
             mock.patch.object(bot_mod.agent, "handle_feishu_event") as h:
            bot._dispatch(evt)
        # Only the current user's image is downloaded; stranger's image
        # is filtered out before consuming budget.
        dl.assert_called_once_with("h_self", "img_user", type_="image")
        hist = h.call_args.kwargs["history"]
        # Stranger's message is also filtered from chat_messages.
        self.assertNotIn("img_other",
                         json.dumps(hist.chat_messages, default=str))


if __name__ == "__main__":
    unittest.main()
