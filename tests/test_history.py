"""Tests for ``app.history.build_history`` and related helpers."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def _msg(
    *,
    message_id: str,
    msg_type: str,
    content: dict,
    sender_open_id: str = "ou_user",
    sender_type: str = "user",
    use_event_shape: bool = False,
):
    m = mock.MagicMock()
    m.message_id = message_id
    m.msg_type = msg_type
    m.body.content = json.dumps(content)
    if use_event_shape:
        # Mirror EventMessage's sender.sender_id.open_id shape — used to
        # verify the fallback path in _extract_sender_open_id.
        m.sender.id = None  # force the fallback
        m.sender.sender_id.open_id = sender_open_id
    else:
        m.sender.id.open_id = sender_open_id
    m.sender.sender_type = sender_type
    return m


class BuildHistoryTests(unittest.TestCase):
    def test_role_mapping_user_and_bot(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="text",
                 content={"text": "hi"}, sender_open_id="ou_user"),
            _msg(message_id="m2", msg_type="text",
                 content={"text": "hello there"}, sender_open_id="ou_bot"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            include_images=False,
        )
        self.assertEqual(len(r.chat_messages), 2)
        self.assertEqual(r.chat_messages[0], {"role": "user", "content": "hi"})
        self.assertEqual(
            r.chat_messages[1],
            {"role": "assistant", "content": "hello there"},
        )

    def test_self_and_bot_filter_drops_third_party(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="text",
                 content={"text": "u"}, sender_open_id="ou_user"),
            _msg(message_id="m2", msg_type="text",
                 content={"text": "other group member"},
                 sender_open_id="ou_someone_else"),
            _msg(message_id="m3", msg_type="text",
                 content={"text": "b"}, sender_open_id="ou_bot"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            include_images=False,
        )
        self.assertEqual(len(r.chat_messages), 2)
        self.assertEqual([m["content"] for m in r.chat_messages], ["u", "b"])

    def test_self_only_filter(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="text",
                 content={"text": "u"}, sender_open_id="ou_user"),
            _msg(message_id="m2", msg_type="text",
                 content={"text": "b"}, sender_open_id="ou_bot"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            filter_mode="self_only",
            include_images=False,
        )
        self.assertEqual(len(r.chat_messages), 1)
        self.assertEqual(r.chat_messages[0]["content"], "u")

    def test_all_users_filter_keeps_strangers(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="text",
                 content={"text": "stranger"},
                 sender_open_id="ou_someone_else"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            filter_mode="all_users",
            include_images=False,
        )
        self.assertEqual(len(r.chat_messages), 1)
        self.assertEqual(r.chat_messages[0]["content"], "stranger")

    def test_skip_message_ids_drops_current_turn(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m_old", msg_type="text",
                 content={"text": "old"}, sender_open_id="ou_user"),
            _msg(message_id="m_current", msg_type="text",
                 content={"text": "current"}, sender_open_id="ou_user"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            skip_message_ids={"m_current"},
            include_images=False,
        )
        contents = [m["content"] for m in r.chat_messages]
        self.assertIn("old", contents)
        self.assertNotIn("current", contents)

    def test_text_only_when_multimodal_disabled(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="image",
                 content={"image_key": "img_a"}, sender_open_id="ou_user"),
            _msg(message_id="m2", msg_type="post",
                 content={"content": [[
                     {"tag": "text", "text": "caption"},
                     {"tag": "img", "image_key": "img_b"},
                 ]]}, sender_open_id="ou_user"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            include_images=False,
            image_bytes_by_key={"img_a": b"PNGDATA"},  # ignored — multimodal off
        )
        # Pure-image message becomes "[image]" placeholder.
        self.assertEqual(r.chat_messages[0],
                         {"role": "user", "content": "[image]"})
        # post: text retained, image dropped.
        self.assertEqual(r.chat_messages[1],
                         {"role": "user", "content": "caption"})
        # All images counted in image_count_skipped_no_multimodal.
        self.assertEqual(r.image_count_total, 2)
        self.assertEqual(r.image_count_included, 0)
        self.assertEqual(r.image_count_skipped_no_multimodal, 2)

    def test_multimodal_inclusion_uses_data_uri(self):
        from app.history import build_history

        png_signature = b"\x89PNG\r\n\x1a\n" + b"X" * 4
        msgs = [
            _msg(message_id="m1", msg_type="image",
                 content={"image_key": "img_a"}, sender_open_id="ou_user"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            include_images=True,
            image_bytes_by_key={"img_a": png_signature},
        )
        self.assertEqual(len(r.chat_messages), 1)
        content = r.chat_messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "image_url")
        url = content[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(r.image_count_included, 1)
        self.assertEqual(r.image_count_skipped_no_multimodal, 0)

    def test_max_images_caps_inclusion(self):
        from app.history import build_history

        png = b"\x89PNG\r\n\x1a\n" + b"P" * 4
        msgs = [
            _msg(message_id=f"m{i}", msg_type="image",
                 content={"image_key": f"img_{i}"}, sender_open_id="ou_user")
            for i in range(5)
        ]
        bytes_map = {f"img_{i}": png for i in range(5)}
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            include_images=True,
            max_images=2,
            image_bytes_by_key=bytes_map,
        )
        # 5 messages total, but only 2 image attachments make the cut. The
        # remaining 3 are simply not embedded (they're not "skipped due to
        # non-multimodal" — that bucket is reserved for the env-flag case).
        included = sum(
            1
            for m in r.chat_messages
            if isinstance(m["content"], list)
            for part in m["content"]
            if part.get("type") == "image_url"
        )
        self.assertEqual(included, 2)
        self.assertEqual(r.image_count_included, 2)
        # Messages whose image was dropped due to cap fall through with
        # role/content present (as "[image]" placeholders) but no image_url.
        self.assertGreaterEqual(len(r.chat_messages), 2)

    def test_missing_image_bytes_drops_image_silently(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="image",
                 content={"image_key": "img_a"}, sender_open_id="ou_user"),
        ]
        # T2T_MULTIMODAL is on, but the bot failed to download img_a.
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            include_images=True,
            image_bytes_by_key={},
        )
        # No multimodal part survives, and we don't count it as
        # "skipped_no_multimodal" because that bucket means the operator
        # disabled multimodal — not a missing-bytes failure.
        self.assertEqual(r.image_count_included, 0)
        self.assertEqual(r.image_count_skipped_no_multimodal, 0)
        # The message itself still exists as a content-less drop — verify
        # no garbage made it into chat_messages.
        for m in r.chat_messages:
            if isinstance(m["content"], list):
                self.assertNotIn(
                    "image_url",
                    [p.get("type") for p in m["content"]],
                )

    def test_text_summary_uses_image_placeholders(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="text",
                 content={"text": "hi"}, sender_open_id="ou_user"),
            _msg(message_id="m2", msg_type="image",
                 content={"image_key": "img_a"}, sender_open_id="ou_user"),
            _msg(message_id="m3", msg_type="text",
                 content={"text": "ok"}, sender_open_id="ou_bot"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            include_images=False,
        )
        self.assertIn("[user]: hi", r.text_summary)
        self.assertIn("[user]: <image>", r.text_summary)
        self.assertIn("[assistant]: ok", r.text_summary)

    def test_empty_messages_returns_empty_result(self):
        from app.history import build_history

        r = build_history(
            [],
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
        )
        self.assertEqual(r.chat_messages, [])
        self.assertEqual(r.text_summary, "")
        self.assertFalse(r.has_content)

    def test_unknown_filter_mode_falls_back_to_self_and_bot(self):
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="text",
                 content={"text": "u"}, sender_open_id="ou_user"),
            _msg(message_id="m2", msg_type="text",
                 content={"text": "?"}, sender_open_id="ou_someone_else"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            filter_mode="nonsense_mode",
            include_images=False,
        )
        self.assertEqual([m["content"] for m in r.chat_messages], ["u"])

    def test_event_shape_sender_fallback(self):
        from app.history import build_history

        # Mimic the EventMessage shape (sender.sender_id.open_id) — the
        # build_history helper falls back to it when sender.id.open_id is
        # missing. We don't normally pass EventMessages to build_history,
        # but the resilience is cheap insurance against SDK churn.
        msgs = [
            _msg(message_id="m1", msg_type="text", content={"text": "u"},
                 sender_open_id="ou_user", use_event_shape=True),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id="ou_bot",
            include_images=False,
        )
        self.assertEqual(r.chat_messages[0],
                         {"role": "user", "content": "u"})

    def test_bot_detection_falls_back_to_sender_type_app(self):
        """When ``FEISHU_BOT_OPEN_ID`` is unset (``bot_open_id=None``),
        ``self_and_bot`` mode still recognizes bot replies as long as
        Feishu tags them with ``sender_type == "app"``. This matters in
        P2P chats — the only foreign sender is the bot itself."""
        from app.history import build_history

        msgs = [
            _msg(message_id="m1", msg_type="text", content={"text": "u"},
                 sender_open_id="ou_user", sender_type="user"),
            _msg(message_id="m2", msg_type="text", content={"text": "b"},
                 sender_open_id="ou_some_app_open_id", sender_type="app"),
            _msg(message_id="m3", msg_type="text", content={"text": "stranger"},
                 sender_open_id="ou_else", sender_type="user"),
        ]
        r = build_history(
            msgs,
            current_user_open_id="ou_user",
            bot_open_id=None,
            include_images=False,
        )
        # Bot reply ("b") is kept via sender_type fallback; "stranger" filtered.
        self.assertEqual([m["content"] for m in r.chat_messages], ["u", "b"])
        self.assertEqual(r.chat_messages[1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
