"""Unit tests for the LLM payload shapes, the router, the Feishu SDK
wrappers, and the agent dispatch — all run offline (HTTP and SDK calls
are monkeypatched).
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# LLM endpoint payload shapes
# ---------------------------------------------------------------------------

def _stub_response(json_body=None, status=200, content=b""):
    r = mock.MagicMock()
    r.status_code = status
    r.text = json.dumps(json_body) if json_body is not None else ""
    r.json.return_value = json_body if json_body is not None else {}
    r.content = content
    return r


def _images_api_response(png_bytes=b"\x89PNG\r\n\x1a\nFAKEDATA"):
    """Build a minimal OpenAI Images-API style response with b64_json."""
    return {
        "data": [
            {"b64_json": base64.b64encode(png_bytes).decode()}
        ]
    }


class ToolPayloadShapeTests(unittest.TestCase):
    # Env vars that might leak in from the user's shell and silently flip
    # config priority on us. We pop them in setUp and restore in tearDown so
    # every test sees a clean baseline regardless of how the suite is invoked.
    _LEAKABLE = (
        "OPENAI_API_BASE",
        "OPENAI_MODEL_NAME",
        "OPENAI_API_VERSION",
        "OPENAI_API_TYPE",
        "OPENAI_DEPLOYMENT_NAME",
        "GPT_TEMPERATURE",
        "T2T_OPENAI_BASE_URL",
        "T2T_OPENAI_API_KEY",
        "T2T_OPENAI_MODEL",
        "T2T_TEMPERATURE",
        "LLM_ENABLE_THINKING",
    )

    def setUp(self):
        for k in self._LEAKABLE:
            prev = os.environ.pop(k, None)
            if prev is not None:
                self.addCleanup(os.environ.__setitem__, k, prev)
        self.env_patch = mock.patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
                "OPENAI_API_KEY": "dummy",
                "OPENAI_MODEL": "model",
                "T2T_OPENAI_BASE_URL": "http://127.0.0.1:8889/v1",
                "T2T_OPENAI_API_KEY": "test-t2t",
                "T2T_OPENAI_MODEL": "t2t-model",
            },
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    # ---- t2t (text chat-completions endpoint) ---------------------------

    def test_t2t_payload(self):
        from app import tools

        captured = {}

        def fake_post(url, *args, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return _stub_response({"choices": [{"message": {"content": "hi"}}]})

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            out = tools.t2t("ping")
        self.assertEqual(out, "hi")
        # Hits the dedicated text endpoint, not the multimodal one.
        self.assertEqual(captured["url"], "http://127.0.0.1:8889/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-t2t")
        body = captured["json"]
        self.assertEqual(body["model"], "t2t-model")
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "ping"})
        # ``modalities`` is kept on text modes so callers who haven't split
        # endpoints yet still coerce vllm-omni into text output.
        self.assertEqual(body["modalities"], ["text"])
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

    def test_t2t_legacy_env_fallback(self):
        """Without T2T_OPENAI_BASE_URL set, fall back to ``OPENAI_API_BASE``
        / ``OPENAI_MODEL_NAME`` (the langchain-style names many users
        already export)."""
        from app import tools

        os.environ.pop("T2T_OPENAI_BASE_URL", None)
        os.environ.pop("T2T_OPENAI_MODEL", None)
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_BASE": "http://legacy.example/v1",
                "OPENAI_MODEL_NAME": "legacy-model",
            },
            clear=False,
        ):
            captured = {}

            def fake_post(url, *args, **kwargs):
                captured["url"] = url
                captured["json"] = kwargs.get("json")
                return _stub_response({"choices": [{"message": {"content": "ok"}}]})

            with mock.patch.object(tools.requests, "post", side_effect=fake_post):
                tools.t2t("ping")
        self.assertEqual(captured["url"], "http://legacy.example/v1/chat/completions")
        self.assertEqual(captured["json"]["model"], "legacy-model")

    def test_t2t_temperature_from_env(self):
        """``GPT_TEMPERATURE`` (legacy name) is honored when no explicit
        temperature is passed and ``T2T_TEMPERATURE`` is unset."""
        from app import tools

        with mock.patch.dict(os.environ, {"GPT_TEMPERATURE": "0.42"}, clear=False):
            captured = {}

            def fake_post(url, *args, **kwargs):
                captured["json"] = kwargs.get("json")
                return _stub_response({"choices": [{"message": {"content": "ok"}}]})

            with mock.patch.object(tools.requests, "post", side_effect=fake_post):
                tools.t2t("ping")
        self.assertAlmostEqual(captured["json"]["temperature"], 0.42)

    def test_t2t_explicit_temperature_overrides_env(self):
        """``agent._llm_route`` passes ``temperature=0.0`` to coerce a
        deterministic routing response — the env fallback must not clobber
        an explicit value (even 0.0)."""
        from app import tools

        with mock.patch.dict(os.environ, {"T2T_TEMPERATURE": "0.9"}, clear=False):
            captured = {}

            def fake_post(url, *args, **kwargs):
                captured["json"] = kwargs.get("json")
                return _stub_response({"choices": [{"message": {"content": "ok"}}]})

            with mock.patch.object(tools.requests, "post", side_effect=fake_post):
                tools.t2t("ping", temperature=0.0)
        self.assertEqual(captured["json"]["temperature"], 0.0)

    def test_extract_text_strips_thinking_block(self):
        from app import tools

        def fake_post(url, *args, **kwargs):
            return _stub_response({
                "choices": [{
                    "message": {
                        "content": "<think>reason about it\nstep by step</think>\n\nFinal answer."
                    }
                }]
            })

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            self.assertEqual(tools.t2t("anything"), "Final answer.")

    def test_extract_text_keeps_untagged_response(self):
        from app import tools

        def fake_post(url, *args, **kwargs):
            return _stub_response({
                "choices": [{"message": {"content": "Plain answer."}}]
            })

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            self.assertEqual(tools.t2t("anything"), "Plain answer.")

    def test_extract_text_preserves_truncated_thinking(self):
        """If the model is cut off mid-thinking (no `</think>`), return the
        original text so the caller still sees *something* rather than an
        empty string."""
        from app import tools

        def fake_post(url, *args, **kwargs):
            return _stub_response({
                "choices": [{"message": {"content": "<think>cut off..."}}]
            })

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            self.assertEqual(tools.t2t("anything"), "<think>cut off...")

    # ---- it2t (text chat-completions endpoint, vision-style message) ----

    def test_it2t_payload(self):
        from app import tools

        img_path = ROOT / "tests_tmp_in2.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0BIN")
        self.addCleanup(img_path.unlink)

        captured = {}

        def fake_post(url, *args, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return _stub_response({"choices": [{"message": {"content": "a cat"}}]})

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            out = tools.it2t(str(img_path), "describe")
        self.assertEqual(out, "a cat")
        # VQA shares the T2T endpoint per the "vqa 和 t2t 用一个模型" directive.
        self.assertEqual(captured["url"], "http://127.0.0.1:8889/v1/chat/completions")
        body = captured["json"]
        self.assertEqual(body["model"], "t2t-model")
        msg = body["messages"][-1]
        self.assertEqual(msg["role"], "user")
        kinds = [p.get("type") for p in msg["content"]]
        self.assertIn("image_url", kinds)
        self.assertIn("text", kinds)
        # Keep modalities=["text"] for safe fallback on omni-style servers.
        self.assertEqual(body["modalities"], ["text"])
        # Carry through temperature/max_tokens (previously dropped on the floor).
        self.assertIn("temperature", body)
        self.assertIn("max_tokens", body)

    # ---- t2i (Images API: /v1/images/generations) -----------------------

    def test_t2i_uses_images_generations(self):
        from app import tools

        png = b"\x89PNG\r\n\x1a\nGEN"
        captured = {}

        def fake_post(url, *args, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return _stub_response(_images_api_response(png))

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            data = tools.t2i("a cat", seed=7)
        self.assertEqual(data, png)
        # Images API endpoint on the *multimodal* base URL.
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/v1/images/generations")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer dummy")
        body = captured["json"]
        self.assertEqual(body["prompt"], "a cat")
        self.assertEqual(body["seed"], 7)
        # Default 1:1 / 1K aspect → 1024x1024 (matches t2i_example.sh).
        self.assertEqual(body["size"], "1024x1024")
        # No chat-completions cruft.
        self.assertNotIn("messages", body)
        self.assertNotIn("modalities", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_t2i_explicit_size_overrides_aspect_ratio(self):
        from app import tools

        captured = {}

        def fake_post(url, *args, **kwargs):
            captured["json"] = kwargs.get("json")
            return _stub_response(_images_api_response())

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            tools.t2i("a cat", size="512x768", aspect_ratio="16:9")
        self.assertEqual(captured["json"]["size"], "512x768")

    def test_t2i_decodes_http_url_response(self):
        """Images API can return a remote ``url`` instead of ``b64_json`` —
        the client must fetch it transparently."""
        from app import tools

        captured = {}

        def fake_post(url, *args, **kwargs):
            return _stub_response({"data": [{"url": "http://x/img.png"}]})

        def fake_get(url, *args, **kwargs):
            captured["fetched"] = url
            return _stub_response(content=b"PNGFETCHED")

        with mock.patch.object(tools.requests, "post", side_effect=fake_post), \
             mock.patch.object(tools.requests, "get", side_effect=fake_get):
            data = tools.t2i("draw")
        self.assertEqual(data, b"PNGFETCHED")
        self.assertEqual(captured["fetched"], "http://x/img.png")

    def test_t2i_http_error_raises(self):
        from app import tools

        def fake_post(*args, **kwargs):
            return _stub_response({"error": "boom"}, status=502)

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            with self.assertRaises(RuntimeError):
                tools.t2i("draw")

    # ---- it2i (Images API: /v1/images/edits, multipart) -----------------

    def test_it2i_uses_images_edits(self):
        from app import tools

        img_path = ROOT / "tests_tmp_in.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0FAKEJPG")
        self.addCleanup(img_path.unlink)

        png = b"\x89PNG\r\n\x1a\nEDIT"
        captured = {}

        def fake_post(url, *args, **kwargs):
            captured["url"] = url
            captured["files"] = kwargs.get("files")
            captured["data"] = kwargs.get("data")
            captured["headers"] = kwargs.get("headers")
            return _stub_response(_images_api_response(png))

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            out = tools.it2i(str(img_path), "make watercolor")
        self.assertEqual(out, png)
        # Multipart Images-Edits endpoint, on the multimodal base URL.
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/v1/images/edits")
        # multipart 'image' part: (filename, bytes, mime)
        image_part = captured["files"]["image"]
        self.assertEqual(image_part[0], img_path.name)
        self.assertEqual(image_part[1], b"\xff\xd8\xff\xe0FAKEJPG")
        self.assertTrue(image_part[2].startswith("image/"))
        # Form fields match it2i_example.sh.
        form = captured["data"]
        self.assertEqual(form["prompt"], "make watercolor")
        self.assertEqual(form["size"], "1024x1024")
        self.assertEqual(form["output_format"], "png")
        # No JSON body for multipart.
        # (requests would set Content-Type from the multipart boundary;
        #  we don't pre-set Content-Type ourselves on this path.)

    def test_it2i_accepts_raw_bytes(self):
        from app import tools

        captured = {}

        def fake_post(url, *args, **kwargs):
            captured["files"] = kwargs.get("files")
            return _stub_response(_images_api_response())

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            tools.it2i(b"\xff\xd8\xff\xe0RAWBYTES", "tweak")
        image_part = captured["files"]["image"]
        # Default filename for raw-byte inputs.
        self.assertTrue(image_part[0].endswith(".jpg"))
        self.assertEqual(image_part[1], b"\xff\xd8\xff\xe0RAWBYTES")

    def test_it2i_http_error_raises(self):
        from app import tools

        img_path = ROOT / "tests_tmp_in_err.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0X")
        self.addCleanup(img_path.unlink)

        def fake_post(*args, **kwargs):
            return _stub_response({"error": "boom"}, status=500)

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            with self.assertRaises(RuntimeError):
                tools.it2i(str(img_path), "tweak")

    # ---- aspect-ratio helper -------------------------------------------

    def test_aspect_ratio_size_resolution(self):
        from app import tools

        # Verbatim size string wins.
        self.assertEqual(tools._resolve_size("800x600", "1:1", "1K"), "800x600")
        # Known preset lookup.
        self.assertEqual(tools._resolve_size(None, "1:1", "1K"), "1024x1024")
        self.assertEqual(tools._resolve_size(None, "16:9", "2K"), "2720x1536")
        self.assertEqual(tools._resolve_size(None, "9:16", "2K"), "1536x2720")
        # Unknown combinations fall back to the example-script default.
        self.assertEqual(tools._resolve_size(None, "21:9", "5K"), "1024x1024")
        self.assertEqual(tools._resolve_size(None, "1:1", "5K"), "1024x1024")

    # ---- chat-completions error path -----------------------------------

    def test_llm_http_error_raises(self):
        from app import tools

        def fake_post(*args, **kwargs):
            return _stub_response({"error": "boom"}, status=502)

        with mock.patch.object(tools.requests, "post", side_effect=fake_post):
            with self.assertRaises(RuntimeError):
                tools.t2t("ping")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class RouterTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "0"}, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_heuristic_no_image_text(self):
        from app import agent
        self.assertEqual(agent.decide_tool("hello", has_image=False), "t2t")

    def test_heuristic_no_image_draw(self):
        from app import agent
        self.assertEqual(agent.decide_tool("please draw a dragon", has_image=False), "t2i")
        self.assertEqual(agent.decide_tool("画一只猫", has_image=False), "t2i")

    def test_heuristic_with_image_describe(self):
        from app import agent
        self.assertEqual(agent.decide_tool("describe", has_image=True), "it2t")

    def test_heuristic_with_image_edit(self):
        from app import agent
        self.assertEqual(agent.decide_tool("turn this into watercolor", has_image=True), "it2i")
        self.assertEqual(agent.decide_tool("把这张图改成卡通风格", has_image=True), "it2i")

    def test_consistency_guard_image_required(self):
        from app import agent
        with self.assertRaises(ValueError):
            agent.decide_tool("hi", has_image=True, mode="t2t")

    def test_llm_router_opt_in(self):
        from app import agent

        fake_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t2i", "arguments": "{}"},
                    }],
                },
            }],
        }
        captured = {}

        def fake_cc(messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return fake_response

        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", side_effect=fake_cc):
            self.assertEqual(agent.decide_tool("ambiguous", has_image=False), "t2i")
        # The router uses the standard OpenAI tools/function-calling protocol.
        self.assertEqual(captured["kwargs"]["tool_choice"], "auto")
        self.assertEqual(captured["kwargs"]["temperature"], 0.0)
        names = [t["function"]["name"] for t in captured["kwargs"]["tools"]]
        # With no image we only offer t2t / t2i — never the it2* options.
        self.assertEqual(sorted(names), ["t2i", "t2t"])

    def test_llm_router_filters_tools_when_image_present(self):
        from app import agent

        fake_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "it2i", "arguments": "{}"},
                    }],
                },
            }],
        }
        captured = {}

        def fake_cc(messages, **kwargs):
            captured["kwargs"] = kwargs
            return fake_response

        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", side_effect=fake_cc):
            self.assertEqual(agent.decide_tool("做点什么", has_image=True), "it2i")
        names = [t["function"]["name"] for t in captured["kwargs"]["tools"]]
        # With an image we only offer it2t / it2i.
        self.assertEqual(sorted(names), ["it2i", "it2t"])

    def test_llm_router_accepts_response_with_malformed_arguments(self):
        """The agent ignores ``function.arguments`` today, so a malformed
        arguments string must not crash routing — only the function ``name``
        matters."""
        from app import agent

        fake_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "t2i", "arguments": "not-json{{{"},
                    }],
                },
            }],
        }
        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", return_value=fake_response):
            self.assertEqual(agent.decide_tool("ambiguous", has_image=False), "t2i")

    def test_llm_router_prefers_tool_calls_over_content(self):
        """If the response has both ``tool_calls`` and a content blob, only
        ``tool_calls`` is consulted."""
        from app import agent

        fake_response = {
            "choices": [{
                "message": {
                    "content": '{"tool":"t2t"}',  # would route to t2t if read
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "t2i", "arguments": "{}"},
                    }],
                },
            }],
        }
        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", return_value=fake_response):
            # Heuristic for "ambiguous" without image would be t2t; tool_calls
            # wins and gives us t2i.
            self.assertEqual(agent.decide_tool("ambiguous", has_image=False), "t2i")

    def test_llm_router_no_tool_calls_falls_back(self):
        """A response with no tool_calls (regardless of content) falls back to
        the keyword router."""
        from app import agent

        fake_response = {"choices": [{"message": {"content": "garbage"}}]}
        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", return_value=fake_response):
            # Falls back to heuristic, which routes "draw" → t2i.
            self.assertEqual(agent.decide_tool("draw an apple", has_image=False), "t2i")

    def test_llm_router_empty_tool_calls_list_falls_back(self):
        from app import agent

        fake_response = {
            "choices": [{
                "message": {"tool_calls": []},
            }],
        }
        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", return_value=fake_response):
            self.assertEqual(agent.decide_tool("draw an apple", has_image=False), "t2i")

    def test_llm_router_invalid_tool_name_falls_back(self):
        """An unknown function name from the LLM falls through to the
        heuristic router."""
        from app import agent

        fake_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "ocr", "arguments": "{}"},
                    }],
                },
            }],
        }
        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", return_value=fake_response):
            self.assertEqual(agent.decide_tool("draw an apple", has_image=False), "t2i")

    def test_llm_router_cross_modality_tool_falls_back(self):
        """An impossible tool for the current has_image state (e.g. it2i
        when no image is attached) must be treated as a routing miss, not
        silently remapped inside ``_llm_route``."""
        from app import agent

        fake_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "it2i", "arguments": "{}"},
                    }],
                },
            }],
        }
        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", return_value=fake_response):
            # has_image=False makes it2i out-of-set → fall back to heuristic
            # for "draw an apple" → t2i.
            self.assertEqual(agent.decide_tool("draw an apple", has_image=False), "t2i")

    def test_llm_router_malformed_tool_calls_shape_falls_back(self):
        """Defensive parsing: any unexpected response shape returns ``None``
        rather than raising, so routing falls back gracefully."""
        from app import agent

        bad_shapes = [
            {"choices": [{"message": {"tool_calls": "not-a-list"}}]},
            {"choices": [{"message": {"tool_calls": ["not-a-dict"]}}]},
            {"choices": [{"message": {"tool_calls": [{"function": "not-a-dict"}]}}]},
            {"choices": [{"message": {"tool_calls": [{"function": {"name": None}}]}}]},
            {"choices": [{"message": {"tool_calls": [{"function": {}}]}}]},
            {"choices": [{"message": "not-a-dict"}]},
            {"choices": []},
            {},
        ]
        for shape in bad_shapes:
            with self.subTest(shape=str(shape)[:80]):
                with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
                     mock.patch.object(agent.tools, "chat_completion", return_value=shape):
                    self.assertEqual(
                        agent.decide_tool("draw an apple", has_image=False),
                        "t2i",
                    )

    def test_llm_router_exception_falls_back(self):
        from app import agent

        def boom(*args, **kwargs):
            raise RuntimeError("backend down")

        with mock.patch.dict(os.environ, {"ENABLE_LLM_ROUTER": "1"}), \
             mock.patch.object(agent.tools, "chat_completion", side_effect=boom):
            self.assertEqual(agent.decide_tool("draw an apple", has_image=False), "t2i")


# ---------------------------------------------------------------------------
# Feishu SDK wrappers
# ---------------------------------------------------------------------------

def _ok(data=None):
    r = mock.MagicMock()
    r.success.return_value = True
    r.code = 0
    r.msg = "ok"
    r.get_log_id.return_value = "log_xyz"
    r.data = data
    return r


def _fail(code=1, msg="failed"):
    r = mock.MagicMock()
    r.success.return_value = False
    r.code = code
    r.msg = msg
    r.get_log_id.return_value = "log_err"
    r.data = None
    return r


class FeishuSDKTests(unittest.TestCase):
    def setUp(self):
        from app import feishu
        self.env_patch = mock.patch.dict(
            os.environ,
            {"FEISHU_APP_ID": "id", "FEISHU_APP_SECRET": "secret"},
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        feishu.reset_client()
        self.addCleanup(feishu.reset_client)
        # Inject a mock SDK client.
        self.fake_client = mock.MagicMock()
        self.client_patch = mock.patch.object(feishu, "get_client", return_value=self.fake_client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def test_reply_text(self):
        from app import feishu
        self.fake_client.im.v1.message.reply.return_value = _ok()

        feishu.reply_text("om_msg", "hi", uuid="ev1:t2t")

        self.fake_client.im.v1.message.reply.assert_called_once()
        req = self.fake_client.im.v1.message.reply.call_args.args[0]
        self.assertEqual(req.message_id, "om_msg")
        self.assertEqual(req.request_body.msg_type, "text")
        self.assertEqual(json.loads(req.request_body.content), {"text": "hi"})
        self.assertEqual(req.request_body.uuid, "ev1:t2t")

    def test_reply_text_long_uuid_truncated(self):
        from app import feishu
        self.fake_client.im.v1.message.reply.return_value = _ok()
        long_uuid = "x" * 80
        feishu.reply_text("om_msg", "hi", uuid=long_uuid)
        req = self.fake_client.im.v1.message.reply.call_args.args[0]
        self.assertLessEqual(len(req.request_body.uuid), 50)

    def test_reply_image_uploads_then_replies(self):
        from app import feishu
        self.fake_client.im.v1.image.create.return_value = _ok(
            data=mock.MagicMock(image_key="img_xyz")
        )
        self.fake_client.im.v1.message.reply.return_value = _ok()

        feishu.reply_image("om_msg", b"PNGBYTES", uuid="ev1:t2i")

        self.fake_client.im.v1.image.create.assert_called_once()
        upload_req = self.fake_client.im.v1.image.create.call_args.args[0]
        self.assertEqual(upload_req.request_body.image_type, "message")
        # body.image is an IO stream — verify by reading its content.
        stream = upload_req.request_body.image
        self.assertIsInstance(stream, io.BytesIO)
        self.assertEqual(stream.getvalue(), b"PNGBYTES")

        self.fake_client.im.v1.message.reply.assert_called_once()
        reply_req = self.fake_client.im.v1.message.reply.call_args.args[0]
        self.assertEqual(reply_req.request_body.msg_type, "image")
        self.assertEqual(json.loads(reply_req.request_body.content), {"image_key": "img_xyz"})

    def test_send_text_uses_chat_id(self):
        from app import feishu
        self.fake_client.im.v1.message.create.return_value = _ok()
        feishu.send_text("oc_chat", "hello")
        req = self.fake_client.im.v1.message.create.call_args.args[0]
        self.assertEqual(req.receive_id_type, "chat_id")
        self.assertEqual(req.request_body.receive_id, "oc_chat")
        self.assertEqual(req.request_body.msg_type, "text")
        self.assertEqual(json.loads(req.request_body.content), {"text": "hello"})

    def test_download_message_resource_returns_bytes(self):
        from app import feishu
        fake_io = io.BytesIO(b"RAWBYTES")
        resp = _ok()
        resp.file = fake_io
        self.fake_client.im.v1.message_resource.get.return_value = resp

        data = feishu.download_message_resource("om_msg", "img_key", type_="image")

        self.assertEqual(data, b"RAWBYTES")
        req = self.fake_client.im.v1.message_resource.get.call_args.args[0]
        self.assertEqual(req.message_id, "om_msg")
        self.assertEqual(req.file_key, "img_key")
        self.assertEqual(req.type, "image")

    def test_download_message_resource_handles_missing_file(self):
        from app import feishu
        resp = _ok()
        resp.file = None
        self.fake_client.im.v1.message_resource.get.return_value = resp
        with self.assertRaises(feishu.FeishuError):
            feishu.download_message_resource("om_msg", "img_key")

    def test_api_failure_raises_feishu_error(self):
        from app import feishu
        self.fake_client.im.v1.message.reply.return_value = _fail(code=99991663, msg="permission")
        with self.assertRaises(feishu.FeishuError):
            feishu.reply_text("om_msg", "hi")

    def test_upload_image_returns_key(self):
        from app import feishu
        self.fake_client.im.v1.image.create.return_value = _ok(
            data=mock.MagicMock(image_key="img_abc")
        )
        key = feishu.upload_image(b"BYTES")
        self.assertEqual(key, "img_abc")

    def test_upload_image_attaches_filename_for_multipart(self):
        """Bare BytesIO without a `.name` is silently dropped by the SDK's
        multipart serializer (Feishu rejects with 234001). Ensure we always
        attach a filename so the multipart part is well-formed."""
        from app import feishu
        captured = {}

        def fake_create(req):
            captured["file"] = req.request_body.image
            return _ok(data=mock.MagicMock(image_key="img_z"))

        self.fake_client.im.v1.image.create.side_effect = fake_create
        feishu.upload_image(b"\x89PNG\r\n\x1a\nrest")
        bio = captured["file"]
        self.assertTrue(hasattr(bio, "name") and bio.name, "BytesIO must have .name set")
        self.assertTrue(bio.name.endswith(".png"))

    def test_get_message_returns_first_item(self):
        from app import feishu
        item = mock.MagicMock()
        item.msg_type = "image"
        item.body.content = '{"image_key":"img_p"}'
        self.fake_client.im.v1.message.get.return_value = _ok(
            data=mock.MagicMock(items=[item])
        )
        result = feishu.get_message("om_parent")
        self.assertIs(result, item)
        req = self.fake_client.im.v1.message.get.call_args.args[0]
        self.assertEqual(req.message_id, "om_parent")

    def test_get_message_empty_items_raises(self):
        from app import feishu
        self.fake_client.im.v1.message.get.return_value = _ok(
            data=mock.MagicMock(items=[])
        )
        with self.assertRaises(feishu.FeishuError):
            feishu.get_message("om_parent")


class FeishuClientBuildTests(unittest.TestCase):
    """Tests for the lazy client-builder, exercised against the env."""

    def test_missing_credentials_raises(self):
        from app import feishu
        feishu.reset_client()
        with mock.patch.dict(os.environ, {"FEISHU_APP_ID": "", "FEISHU_APP_SECRET": ""},
                             clear=False):
            with self.assertRaises(feishu.FeishuError):
                feishu.get_client()


# ---------------------------------------------------------------------------
# Agent CLI dispatch (run mode, now SDK-backed)
# ---------------------------------------------------------------------------

class AgentRunTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = mock.patch.dict(
            os.environ,
            {
                "ENABLE_LLM_ROUTER": "0",
                "FEISHU_APP_ID": "id",
                "FEISHU_APP_SECRET": "secret",
                "FEISHU_CHAT_ID": "oc_default",
            },
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.outdir = ROOT / "tests_tmp_out"
        self.outdir.mkdir(exist_ok=True)
        self.addCleanup(self._cleanup_outdir)

    def _cleanup_outdir(self):
        for p in self.outdir.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            self.outdir.rmdir()
        except OSError:
            pass

    def test_run_t2t_sends_to_default_chat(self):
        from app import agent
        with mock.patch.object(agent.tools, "t2t", return_value="answer"), \
             mock.patch.object(agent.feishu, "send_text") as st:
            result = agent.run("hello world", output_dir=str(self.outdir))
        self.assertEqual(result.tool, "t2t")
        self.assertEqual(result.text, "answer")
        self.assertTrue(result.feishu_pushed)
        st.assert_called_once()
        chat_id, body = st.call_args.args
        self.assertEqual(chat_id, "oc_default")
        self.assertIn("answer", body)

    def test_run_t2t_explicit_chat_id_overrides_env(self):
        from app import agent
        with mock.patch.object(agent.tools, "t2t", return_value="ok"), \
             mock.patch.object(agent.feishu, "send_text") as st:
            agent.run("hi", chat_id="oc_override", output_dir=str(self.outdir))
        st.assert_called_once()
        self.assertEqual(st.call_args.args[0], "oc_override")

    def test_run_without_chat_id_reports_error(self):
        from app import agent
        with mock.patch.dict(os.environ, {"FEISHU_CHAT_ID": ""}, clear=False), \
             mock.patch.object(agent.tools, "t2t", return_value="ok"), \
             mock.patch.object(agent.feishu, "send_text") as st:
            result = agent.run("hi", output_dir=str(self.outdir))
        st.assert_not_called()
        self.assertFalse(result.feishu_pushed)
        self.assertIn("FEISHU_CHAT_ID", result.feishu_error)

    def test_run_t2i_saves_and_sends_image(self):
        from app import agent
        with mock.patch.object(agent.tools, "t2i", return_value=b"PNGBYTES"), \
             mock.patch.object(agent.feishu, "send_image") as si, \
             mock.patch.object(agent.feishu, "send_text") as st:
            result = agent.run("draw an apple", mode="t2i", output_dir=str(self.outdir))
        self.assertEqual(result.tool, "t2i")
        self.assertTrue(Path(result.image_path).exists())
        self.assertTrue(result.feishu_pushed)
        si.assert_called_once()
        self.assertEqual(si.call_args.args[0], "oc_default")
        self.assertEqual(si.call_args.args[1], b"PNGBYTES")
        # Caption should also be sent.
        st.assert_called_once()
        self.assertIn("draw an apple", st.call_args.args[1])

    def test_run_mode_validation_image_required(self):
        from app import agent
        with self.assertRaises(ValueError):
            agent.run("describe", mode="it2t")

    def test_run_mode_validation_no_image_for_t2t(self):
        from app import agent
        p = ROOT / "tests_tmp_in3.jpg"
        p.write_bytes(b"x")
        self.addCleanup(p.unlink)
        with self.assertRaises(ValueError):
            agent.run("hi", mode="t2t", image_path=str(p))

    def test_run_no_feishu_skips_delivery(self):
        from app import agent
        with mock.patch.object(agent.tools, "t2t", return_value="ok"), \
             mock.patch.object(agent.feishu, "send_text") as st:
            result = agent.run("hello", notify_feishu=False, output_dir=str(self.outdir))
        self.assertEqual(result.text, "ok")
        self.assertFalse(result.feishu_pushed)
        st.assert_not_called()


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class CLITests(unittest.TestCase):
    def test_parser_run_subcommand(self):
        from app import __main__ as cli
        parser = cli.build_parser()
        ns = parser.parse_args(
            ["run", "my prompt", "--mode", "t2t", "--no-feishu", "--chat-id", "oc_x"]
        )
        self.assertEqual(ns.cmd, "run")
        self.assertEqual(ns.prompt, "my prompt")
        self.assertEqual(ns.mode, "t2t")
        self.assertTrue(ns.no_feishu)
        self.assertEqual(ns.chat_id, "oc_x")

    def test_parser_bare_prompt_via_main_shim(self):
        from app import __main__ as cli
        from app import agent

        captured = {}

        def fake_run(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["mode"] = kwargs.get("mode")
            captured["notify"] = kwargs.get("notify_feishu")
            return agent.AgentResult(tool="t2t", text="ok", image_path=None,
                                     feishu_pushed=False)

        with mock.patch("app.__main__.run", side_effect=fake_run):
            rc = cli.main(["my prompt", "--mode", "t2t", "--no-feishu"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["prompt"], "my prompt")
        self.assertEqual(captured["mode"], "t2t")
        self.assertFalse(captured["notify"])

    def test_parser_serve_subcommand(self):
        from app import __main__ as cli
        parser = cli.build_parser()
        ns = parser.parse_args(["serve"])
        self.assertEqual(ns.cmd, "serve")


if __name__ == "__main__":
    unittest.main()
