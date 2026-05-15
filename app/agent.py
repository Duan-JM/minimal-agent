"""Agent layer: route a user request to one of the four capabilities and
deliver the result back to Feishu (either as a one-shot send in CLI mode, or
as a threaded reply in bot mode)."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from . import feishu, tools
from .history import HistoryResult


VALID_TOOLS = ("t2t", "t2i", "it2t", "it2i")

# OpenAI tools/function-calling schema for each capability. The router
# advertises only the schemas compatible with the current has_image value so
# the LLM never sees impossible options. The ``prompt`` argument exists so
# the schema is well-formed (and so a future change can act on a refined
# prompt) — the agent currently ignores the returned arguments and reuses
# the user's original prompt.
_ROUTER_TOOL_SCHEMAS: dict = {
    "t2t": {
        "type": "function",
        "function": {
            "name": "t2t",
            "description": (
                "Answer the user with text only. Use when no image is attached "
                "and the user wants a textual answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The user prompt to answer."}
                },
                "required": ["prompt"],
            },
        },
    },
    "t2i": {
        "type": "function",
        "function": {
            "name": "t2i",
            "description": (
                "Generate a brand-new image from a text description. Use when "
                "no image is attached and the user wants a picture back."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Image-generation prompt (English works best).",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    "it2t": {
        "type": "function",
        "function": {
            "name": "it2t",
            "description": (
                "Describe or answer questions about the user's input image with "
                "text. Use when an image is attached and the user wants a text answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The question or task to perform on the image.",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    "it2i": {
        "type": "function",
        "function": {
            "name": "it2i",
            "description": (
                "Edit / restyle / transform the user's input image and return a "
                "new image. Use when an image is attached and the user wants a "
                "modified picture back."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Edit instruction (English works best).",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
}

IMAGE_GEN_KEYWORDS = (
    "draw", "generate image", "generate an image", "generate a picture",
    "make an image", "make a picture", "picture of", "image of", "render",
    "illustration", "paint",
    "画", "绘", "绘制", "生成图", "生成一张图", "生成图片", "出图", "画一张", "画一幅",
)
IMAGE_EDIT_KEYWORDS = (
    "edit", "turn this", "make this", "convert this", "change this", "restyle",
    "redraw", "transform", "stylize", "make it",
    "编辑", "改成", "变成", "转成", "转换", "改为", "改图", "改一改", "替换",
)

DEFAULT_IMAGE_ONLY_PROMPT = "请用一段话描述这张图片的内容。"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    tool: str
    text: Optional[str] = None
    image_bytes: Optional[bytes] = None
    image_path: Optional[str] = None


@dataclass
class AgentResult:
    """CLI-mode return value."""
    tool: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    feishu_pushed: bool = False
    feishu_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _heuristic_route(prompt: str, has_image: bool) -> str:
    p = (prompt or "").lower()
    if has_image:
        if any(k in p for k in IMAGE_EDIT_KEYWORDS):
            return "it2i"
        if any(k in p for k in IMAGE_GEN_KEYWORDS):
            return "it2i"
        return "it2t"
    if any(k in p for k in IMAGE_GEN_KEYWORDS):
        return "t2i"
    return "t2t"


def _llm_route(prompt: str, has_image: bool) -> Optional[str]:
    """Ask the LLM to pick a tool via OpenAI's tools/function-calling.

    Returns ``None`` (so ``route()`` falls back to the keyword router) when:

    - the chat call raises,
    - the response has no usable ``tool_calls``,
    - the picked tool name isn't one of the offered candidates.

    All response parsing is defensive — any unexpected shape returns ``None``
    rather than raising, so a flaky backend never breaks routing.
    """
    candidate_names = ("it2t", "it2i") if has_image else ("t2t", "t2i")
    tool_schemas = [_ROUTER_TOOL_SCHEMAS[name] for name in candidate_names]

    system_msg = (
        "You are a tool router. Pick exactly one of the available functions "
        "that best matches the user's request. Do not answer the user yourself."
    )
    user_msg = (
        f"User prompt: {prompt!r}\n"
        f"Has image attached: {'yes' if has_image else 'no'}\n"
        "Call the single function that best fits."
    )

    try:
        data = tools.chat_completion(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=64,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[router] LLM routing failed, falling back to heuristic: {e}", file=sys.stderr)
        return None

    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        print(f"[router] LLM response had no message: {str(data)[:300]!r}", file=sys.stderr)
        return None
    if not isinstance(message, dict):
        print(f"[router] LLM message is not a dict: {type(message).__name__}", file=sys.stderr)
        return None

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        print(
            "[router] LLM returned no tool_calls; falling back to heuristic",
            file=sys.stderr,
        )
        return None

    first = tool_calls[0]
    if not isinstance(first, dict):
        print(f"[router] LLM tool_calls[0] is not a dict: {first!r}", file=sys.stderr)
        return None
    fn = first.get("function")
    if not isinstance(fn, dict):
        print(f"[router] LLM tool_calls[0].function is not a dict: {fn!r}", file=sys.stderr)
        return None
    name = fn.get("name")
    if not isinstance(name, str):
        print(f"[router] LLM returned non-string function name: {name!r}", file=sys.stderr)
        return None
    name = name.strip()
    # Validate against the offered candidates — not the full VALID_TOOLS —
    # so a buggy backend returning an impossible cross-modality tool is
    # surfaced as a routing miss rather than silently remapped.
    if name not in candidate_names:
        print(
            f"[router] LLM returned out-of-set tool name: {name!r} "
            f"(allowed: {candidate_names})",
            file=sys.stderr,
        )
        return None
    return name


def route(prompt: str, has_image: bool) -> str:
    tool: Optional[str] = None
    if os.environ.get("ENABLE_LLM_ROUTER", "").strip() in ("1", "true", "True", "yes"):
        tool = _llm_route(prompt, has_image)
    if tool is None:
        tool = _heuristic_route(prompt, has_image)
    # Consistency guard against mismatched (mode, has_image).
    if has_image and tool in ("t2t", "t2i"):
        tool = "it2t" if tool == "t2t" else "it2i"
    if (not has_image) and tool in ("it2t", "it2i"):
        tool = "t2t" if tool == "it2t" else "t2i"
    return tool


def decide_tool(prompt: str, has_image: bool, mode: Optional[str] = None) -> str:
    if mode is not None:
        if mode not in VALID_TOOLS:
            raise ValueError(f"Invalid mode {mode!r}; must be one of {VALID_TOOLS}")
        if has_image and mode in ("t2t", "t2i"):
            raise ValueError(f"mode={mode} does not accept an image input; use it2t or it2i")
        if (not has_image) and mode in ("it2t", "it2i"):
            raise ValueError(f"mode={mode} requires an image input")
        return mode
    return route(prompt, has_image)


def execute_tool(tool: str, prompt: str,
                 image: Optional[Union[str, Path, bytes]] = None,
                 history: Optional[HistoryResult] = None) -> ToolResult:
    chat_history = history.chat_messages if history else None
    text_prefix = history.text_summary if history else None
    if tool == "t2t":
        return ToolResult(tool=tool, text=tools.t2t(prompt, history=chat_history))
    if tool == "it2t":
        if image is None:
            raise ValueError("it2t requires an image input")
        return ToolResult(
            tool=tool, text=tools.it2t(image, prompt, history=chat_history)
        )
    if tool == "t2i":
        return ToolResult(
            tool=tool,
            image_bytes=tools.t2i(prompt, history_text_prefix=text_prefix),
        )
    if tool == "it2i":
        if image is None:
            raise ValueError("it2i requires an image input")
        return ToolResult(
            tool=tool,
            image_bytes=tools.it2i(image, prompt, history_text_prefix=text_prefix),
        )
    raise ValueError(f"unknown tool {tool!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_image(image_bytes: bytes, output_dir: Union[str, Path], tool: str,
               suffix: str = "jpg") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"{ts}_{tool}.{suffix}"
    out.write_bytes(image_bytes)
    return out


def _make_uuid(event_id: Optional[str], message_id: Optional[str], tool: str) -> Optional[str]:
    base = event_id or message_id
    if not base:
        return None
    return f"{base}:{tool}"


# ---------------------------------------------------------------------------
# CLI delivery path (one-shot send via SDK)
# ---------------------------------------------------------------------------

def _resolve_chat_id(explicit: Optional[str]) -> Optional[str]:
    return explicit or os.environ.get("FEISHU_CHAT_ID") or None


def _push_text(chat_id: str, text: str, header: str) -> tuple[bool, Optional[str]]:
    body = f"{header}\n\n{text}" if header else text
    try:
        feishu.send_text(chat_id, body)
        return True, None
    except feishu.FeishuError as e:
        print(f"[feishu] send failed: {e}", file=sys.stderr)
        return False, str(e)


def _push_image(chat_id: str, image_bytes: bytes, prompt: str, tool: str
                ) -> tuple[bool, Optional[str]]:
    try:
        feishu.send_image(chat_id, image_bytes)
    except feishu.FeishuError as e:
        print(f"[feishu] image send failed: {e}", file=sys.stderr)
        return False, str(e)
    try:
        feishu.send_text(chat_id, f"🎨 [{tool}] {prompt}")
    except feishu.FeishuError as e:
        # Image already delivered — caption failure is non-fatal.
        print(f"[feishu] caption send failed (image already sent): {e}", file=sys.stderr)
    return True, None


def run(prompt: str, image_path: Optional[str] = None, mode: Optional[str] = None,
        output_dir: str = "./output_images", notify_feishu: bool = True,
        chat_id: Optional[str] = None) -> AgentResult:
    """Single-shot CLI dispatch: route + run + (optionally) push to a Feishu chat."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    has_image = bool(image_path)
    if has_image and not Path(image_path).is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    tool = decide_tool(prompt, has_image, mode)
    print(f"[agent] tool={tool} has_image={has_image}", file=sys.stderr)

    result = execute_tool(tool, prompt, image_path if has_image else None)

    target_chat = _resolve_chat_id(chat_id) if notify_feishu else None

    if result.text is not None:
        print(result.text)
        pushed, err = (False, None)
        if notify_feishu:
            if not target_chat:
                err = ("FEISHU_CHAT_ID (or --chat-id) is required to deliver "
                       "messages; pass --no-feishu to skip.")
                print(f"[feishu] {err}", file=sys.stderr)
            else:
                header = f"💬 [{tool}] {prompt}" if tool == "t2t" else f"🖼️→💬 [{tool}] {prompt}"
                pushed, err = _push_text(target_chat, result.text, header=header)
        return AgentResult(tool=tool, text=result.text,
                           feishu_pushed=pushed, feishu_error=err)

    assert result.image_bytes is not None
    out = save_image(result.image_bytes, output_dir, tool)
    print(f"[agent] image saved: {out} ({len(result.image_bytes)} bytes)", file=sys.stderr)
    pushed, err = (False, None)
    if notify_feishu:
        if not target_chat:
            err = ("FEISHU_CHAT_ID (or --chat-id) is required to deliver "
                   "messages; pass --no-feishu to skip.")
            print(f"[feishu] {err}", file=sys.stderr)
        else:
            pushed, err = _push_image(target_chat, result.image_bytes, prompt, tool)
    return AgentResult(tool=tool, image_path=str(out),
                       feishu_pushed=pushed, feishu_error=err)


# ---------------------------------------------------------------------------
# Bot delivery path (reply via IM API)
# ---------------------------------------------------------------------------

def handle_feishu_event(
    *,
    text: str,
    image_bytes: Optional[bytes],
    message_id: str,
    chat_id: str,
    event_id: Optional[str] = None,
    mode: Optional[str] = None,
    output_dir: str = "./output_images",
    save_image_locally: bool = True,
    history: Optional[HistoryResult] = None,
) -> ToolResult:
    """Dispatch a parsed Feishu message and reply through the IM API.

    The reply uses ``/im/v1/messages/{message_id}/reply`` so answers thread
    under the user's message. ``event_id`` (if provided) is used as part of the
    message ``uuid`` for idempotency.

    ``history`` (optional): pre-built conversation context from
    :func:`app.history.build_history`. ``chat_messages`` are passed to
    ``t2t`` / ``it2t`` and ``text_summary`` is folded into the prompt for
    ``t2i`` / ``it2i``.
    """
    has_image = image_bytes is not None
    if not text and not has_image:
        raise ValueError("event has neither text nor image; nothing to do")

    prompt = text.strip() if text else DEFAULT_IMAGE_ONLY_PROMPT
    tool = decide_tool(prompt, has_image, mode)
    history_info = ""
    if history and history.has_content:
        history_info = (
            f" history_msgs={len(history.chat_messages)}"
            f" history_imgs={history.image_count_included}"
            f"/{history.image_count_total}"
        )
    print(
        f"[agent] feishu event tool={tool} has_image={has_image} chat_id={chat_id} "
        f"msg={message_id} event={event_id}{history_info}",
        file=sys.stderr,
    )

    result = execute_tool(tool, prompt, image_bytes if has_image else None,
                          history=history)

    uuid = _make_uuid(event_id, message_id, tool)

    if result.text is not None:
        feishu.reply_text(message_id, result.text, uuid=uuid)
        return result

    assert result.image_bytes is not None
    if save_image_locally:
        out = save_image(result.image_bytes, output_dir, tool)
        result.image_path = str(out)
    feishu.reply_image(message_id, result.image_bytes, uuid=uuid)
    return result

