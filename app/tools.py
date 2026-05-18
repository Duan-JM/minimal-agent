"""Four capability tools backed by two endpoints.

Architecture (per ``vqa 和 t2t 用一个模型`` directive):

  text-only chat endpoint  (T2T_* env vars, default qwen3-30b-a3b @ llama.cpp)
    │
    ├── t2t   text  -> text                  (POST /v1/chat/completions)
    └── it2t  image + text -> text  (VQA)    (POST /v1/chat/completions,
                                              vision-style messages — requires
                                              the chat endpoint to be VL-
                                              capable, e.g. llama.cpp with an
                                              mmproj adapter loaded)

  multimodal images endpoint (OPENAI_BASE_URL, default vllm-omni / sense_nova_u1)
    │
    ├── t2i   text  -> image bytes           (POST /v1/images/generations)
    └── it2i  image + text -> image bytes    (POST /v1/images/edits, multipart)

Notes:
- For text modes the model is fine-tuned to emit a leading ``<think>...</think>``
  block; :func:`_extract_text` strips it before returning so callers see only
  the user-facing answer.
- The Images API endpoints match ``t2i_example.sh`` / ``it2i_example.sh``.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests

from .log_config import get_logger


log = get_logger(__name__)


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "model"
DEFAULT_API_KEY = "dummy"
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))

ImageInput = Union[str, Path, bytes]


def _config() -> Dict[str, str]:
    """Config for the multimodal images endpoint (used by t2i / it2i)."""
    return {
        "base_url": os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "api_key": os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY),
        "model": os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
    }


def _t2t_config() -> Dict[str, str]:
    """Config for the text chat-completions endpoint (used by t2t and it2t).

    Resolves with priority::

        base_url:  T2T_OPENAI_BASE_URL > OPENAI_API_BASE   > OPENAI_BASE_URL
        api_key:   T2T_OPENAI_API_KEY  >                     OPENAI_API_KEY
        model:     T2T_OPENAI_MODEL    > OPENAI_MODEL_NAME > OPENAI_MODEL

    The legacy fallback names (``OPENAI_API_BASE`` / ``OPENAI_MODEL_NAME``)
    let users plug in env vars from older clients (langchain-style) without
    renaming. With no T2T_* / legacy names set, t2t reuses the multimodal
    endpoint.
    """
    default = _config()
    base = (
        os.environ.get("T2T_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or default["base_url"]
    )
    api_key = (
        os.environ.get("T2T_OPENAI_API_KEY")
        or default["api_key"]
    )
    model = (
        os.environ.get("T2T_OPENAI_MODEL")
        or os.environ.get("OPENAI_MODEL_NAME")
        or default["model"]
    )
    return {"base_url": base.rstrip("/"), "api_key": api_key, "model": model}


def _t2t_temperature(explicit: Optional[float] = None) -> float:
    """Resolve the t2t/it2t temperature: explicit > T2T_TEMPERATURE >
    GPT_TEMPERATURE > 0.7."""
    if explicit is not None:
        return explicit
    raw = os.environ.get("T2T_TEMPERATURE") or os.environ.get("GPT_TEMPERATURE")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 0.7


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _post_chat(
    payload: Dict[str, Any], *, cfg: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    if cfg is None:
        cfg = _config()
    url = f"{cfg['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    log.debug(
        "tools.chat.request",
        url=url,
        model=payload.get("model"),
        messages_count=len(payload.get("messages") or []),
        has_tools=bool(payload.get("tools")),
        max_tokens=payload.get("max_tokens"),
        temperature=payload.get("temperature"),
        stream=payload.get("stream"),
    )
    started = time.monotonic()
    resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if resp.status_code >= 400:
        log.error(
            "tools.chat.http_error",
            url=url,
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
            body_preview=resp.text[:500],
        )
        raise RuntimeError(
            f"LLM HTTP {resp.status_code} from {url}: {resp.text[:500]}"
        )
    try:
        data = resp.json()
    except ValueError as e:
        log.error(
            "tools.chat.non_json_response",
            url=url,
            elapsed_ms=elapsed_ms,
            body_preview=resp.text[:500],
        )
        raise RuntimeError(f"LLM returned non-JSON response: {resp.text[:500]}") from e
    if isinstance(data, dict) and data.get("error"):
        log.error(
            "tools.chat.api_error",
            url=url,
            elapsed_ms=elapsed_ms,
            error=data["error"],
        )
        raise RuntimeError(f"LLM API error: {data['error']}")
    log.debug(
        "tools.chat.response",
        url=url,
        status=resp.status_code,
        elapsed_ms=elapsed_ms,
        response_bytes=len(resp.content),
        choices=len(data.get("choices") or []) if isinstance(data, dict) else 0,
    )
    return data


def _encode_image_data_uri(image: ImageInput) -> str:
    """Encode an image (path or raw bytes) as a ``data:`` URI."""
    if isinstance(image, (bytes, bytearray)):
        return f"data:image/jpeg;base64,{base64.b64encode(bytes(image)).decode()}"
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Strip any leading ``<think>...</think>`` block(s).

    The ``sense_nova_u1`` chat template always emits a leading reasoning
    block; only what comes after ``</think>`` is meant for the user. If
    no closing tag is present (e.g. truncated response) we return the
    original text untouched so the caller still sees *something*.
    """
    if "<think>" not in text:
        return text.strip()
    if "</think>" not in text:
        return text.strip()
    return _THINK_RE.sub("", text).strip()


def _extract_text(data: Dict[str, Any]) -> str:
    try:
        choices = data["choices"]
        msg = choices[0]["message"]
        content = msg.get("content")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected LLM response shape: {json.dumps(data)[:500]}") from e
    if content is None:
        raise RuntimeError(f"LLM response had no text content: {json.dumps(data)[:500]}")
    if isinstance(content, list):
        # Some servers return a list of content parts.
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return _strip_thinking("".join(parts))
    return _strip_thinking(str(content))


def _decode_b64_image(data: Dict[str, Any]) -> bytes:
    """Decode the first image from an OpenAI Images-API style response.

    Handles ``b64_json`` (the local default), ``url`` data URIs, and remote
    ``http(s)`` URLs as fallbacks.
    """
    items = data.get("data") or []
    if not items or not isinstance(items[0], dict):
        raise RuntimeError(f"No image in Images API response: {json.dumps(data)[:500]}")
    first = items[0]
    b64 = first.get("b64_json")
    if b64:
        return base64.b64decode(b64)
    url = first.get("url") or ""
    if url.startswith("data:"):
        try:
            _, raw_b64 = url.split(",", 1)
        except ValueError as e:
            raise RuntimeError(f"Malformed data URI: {url[:80]}") from e
        return base64.b64decode(raw_b64)
    if url.startswith(("http://", "https://")):
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError(f"Failed to fetch image at {url}: {r.status_code}")
        return r.content
    raise RuntimeError(f"No usable image in response item: {json.dumps(first)[:500]}")


def _image_to_multipart(image: ImageInput) -> Tuple[str, bytes, str]:
    """Convert ``ImageInput`` (path / bytes) to a multipart upload tuple."""
    if isinstance(image, (bytes, bytearray)):
        return ("input.jpg", bytes(image), "image/jpeg")
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    return (path.name, path.read_bytes(), mime)


# Aspect-ratio -> (width, height) presets, mirroring
# ``official_example._aspect_ratio_to_resolution``.
_RES_MAP: Dict[str, Dict[str, Tuple[int, int]]] = {
    "1:1":  {"1K": (1024, 1024), "1.5K": (1536, 1536), "2K": (2048, 2048)},
    "16:9": {"1.5K": (2048, 1152), "2K": (2720, 1536)},
    "9:16": {"1.5K": (1152, 2048), "2K": (1536, 2720)},
    "3:2":  {"1.5K": (1888, 1248), "2K": (2496, 1664)},
    "2:3":  {"1.5K": (1248, 1888), "2K": (1664, 2496)},
    "4:3":  {"1.5K": (1760, 1312), "2K": (2368, 1760)},
    "3:4":  {"1.5K": (1312, 1760), "2K": (1760, 2368)},
}


def _resolve_size(
    size: Optional[str], aspect_ratio: str, image_size: str
) -> str:
    """Resolve an OpenAI Images-API ``size`` string (``"WxH"``).

    If ``size`` is given verbatim, return it. Otherwise look up
    ``aspect_ratio`` + ``image_size`` in :data:`_RES_MAP`, falling back to
    ``"1024x1024"`` (matches the example scripts) when there is no match.
    """
    if size:
        return size
    wh = (_RES_MAP.get(aspect_ratio) or {}).get(image_size)
    if wh:
        return f"{wh[0]}x{wh[1]}"
    return "1024x1024"


def _wrap_with_history(prompt: str, history_text_prefix: Optional[str]) -> str:
    """Prepend conversation context to an Images-API prompt.

    The Images API has no notion of multi-turn messages, so the bot
    pre-renders recent chat history as plain text and we paste it in
    front of the user's request, framed so the model can tell what's
    context vs. what to draw. When ``history_text_prefix`` is empty or
    None we return ``prompt`` unchanged so non-bot callers see no diff.
    """
    if not history_text_prefix:
        return prompt
    return (
        f"Conversation context (oldest -> newest):\n"
        f"{history_text_prefix}\n\n"
        f"Current request: {prompt}"
    )


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

DEFAULT_T2T_SYSTEM = "You are a helpful assistant."


def chat_completion(
    messages: list,
    *,
    tools: Optional[list] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Low-level chat-completions call returning the raw response dict.

    Targets the same ``T2T_*`` endpoint as :func:`t2t` / :func:`it2t` and
    keeps the ``modalities=["text"]`` + ``enable_thinking`` template kwargs
    consistent. The router uses this to invoke OpenAI's tools /
    function-calling protocol and inspect ``choices[0].message.tool_calls``.

    Pass ``tools=[…]`` to enable function-calling. ``tool_choice`` is forwarded
    verbatim (``"auto"``, ``"required"``, or ``{"type": "function", ...}``).
    """
    cfg = _t2t_config()
    payload: Dict[str, Any] = {
        "model": cfg["model"],
        "messages": messages,
        "modalities": ["text"],
        "stream": False,
        "temperature": _t2t_temperature(temperature),
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": _env_bool("LLM_ENABLE_THINKING", False)
        },
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return _post_chat(payload, cfg=cfg)


def t2t(
    prompt: str,
    system: str = DEFAULT_T2T_SYSTEM,
    temperature: Optional[float] = None,
    max_tokens: int = 4096,
    history: Optional[list] = None,
) -> str:
    """Text -> text.

    Targets the dedicated text chat-completions endpoint configured via
    ``T2T_*`` (default qwen3-30b-a3b on llama.cpp). ``modalities=["text"]``
    is kept for safety so that callers who haven't split their endpoints
    yet still get text from a vllm-omni server.

    ``history`` (optional): a list of prior OpenAI-format chat messages
    (each ``{"role": "user"|"assistant", "content": str | list}``) to
    insert between the system message and the current user prompt. The
    bot passes the conversation history built by :func:`app.history.build_history`.
    """
    messages: list = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    data = chat_completion(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _extract_text(data)


def it2t(
    image: ImageInput,
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 4096,
    history: Optional[list] = None,
) -> str:
    """Image + text -> text (VQA).

    Shares the same chat-completions endpoint as :func:`t2t` per the
    "vqa 和 t2t 用一个模型" directive — i.e. it goes to the ``T2T_*``
    endpoint, not the images endpoint.

    ``history`` (optional): inserted between the optional system message
    and the current user turn (which carries the image + prompt). Image
    parts inside ``history`` require a multimodal-capable T2T endpoint
    — see ``T2T_MULTIMODAL`` in the README.

    .. warning::
        The default qwen3-30b-a3b deployment on llama.cpp is text-only
        (no ``mmproj`` adapter loaded) and will respond with HTTP 500
        ``"image input is not supported"`` when an image is passed.
        Either load an mmproj adapter on the llama.cpp server or point
        ``T2T_OPENAI_BASE_URL`` at a VL-capable chat endpoint.
    """
    cfg = _t2t_config()
    url = _encode_image_data_uri(image)
    messages: list = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": prompt},
            ],
        }
    )
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "modalities": ["text"],
        "stream": False,
        "temperature": _t2t_temperature(temperature),
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": _env_bool("LLM_ENABLE_THINKING", False)
        },
    }
    return _extract_text(_post_chat(payload, cfg=cfg))


def t2i(
    prompt: str,
    *,
    size: Optional[str] = None,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    seed: int = 42,
    history_text_prefix: Optional[str] = None,
) -> bytes:
    """Text -> image bytes via the OpenAI Images API.

    Mirrors ``t2i_example.sh``: ``POST {base_url}/images/generations`` with
    ``{prompt, size, seed}``. The response is parsed for ``data[0].b64_json``
    (with ``url`` fallbacks).

    ``history_text_prefix`` (optional): a plaintext rendering of recent
    conversation context. The Images API doesn't accept multi-turn
    messages, so we fold history into the prompt string by prepending it
    inside a ``Context: ... Request: ...`` block. The bot computes this
    via :attr:`app.history.HistoryResult.text_summary`.
    """
    cfg = _config()
    url = f"{cfg['base_url']}/images/generations"
    final_prompt = _wrap_with_history(prompt, history_text_prefix)
    body = {
        "prompt": final_prompt,
        "size": _resolve_size(size, aspect_ratio, image_size),
        "seed": seed,
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    log.debug(
        "tools.t2i.request",
        url=url,
        size=body["size"],
        seed=seed,
        prompt_chars=len(final_prompt),
        has_history=bool(history_text_prefix),
    )
    started = time.monotonic()
    r = requests.post(url, json=body, headers=headers, timeout=DEFAULT_TIMEOUT)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if r.status_code >= 400:
        log.error(
            "tools.t2i.http_error",
            url=url,
            status=r.status_code,
            elapsed_ms=elapsed_ms,
            body_preview=r.text[:500],
        )
        raise RuntimeError(
            f"Images API HTTP {r.status_code} from {url}: {r.text[:500]}"
        )
    try:
        data = r.json()
    except ValueError as e:
        log.error(
            "tools.t2i.non_json_response",
            url=url,
            elapsed_ms=elapsed_ms,
            body_preview=r.text[:500],
        )
        raise RuntimeError(f"Images API non-JSON response: {r.text[:500]}") from e
    if isinstance(data, dict) and data.get("error"):
        log.error(
            "tools.t2i.api_error",
            url=url,
            elapsed_ms=elapsed_ms,
            error=data["error"],
        )
        raise RuntimeError(f"Images API error: {data['error']}")
    image_bytes = _decode_b64_image(data)
    log.debug(
        "tools.t2i.response",
        url=url,
        status=r.status_code,
        elapsed_ms=elapsed_ms,
        image_bytes=len(image_bytes),
    )
    return image_bytes


def it2i(
    image: ImageInput,
    prompt: str,
    *,
    size: Optional[str] = None,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    output_format: str = "png",
    history_text_prefix: Optional[str] = None,
) -> bytes:
    """Image + text -> image bytes via the OpenAI Images Edits API.

    Mirrors ``it2i_example.sh``: ``POST {base_url}/images/edits`` with a
    multipart body (``image`` file part + ``prompt`` / ``size`` /
    ``output_format`` form fields). Response parsing matches :func:`t2i`.

    ``history_text_prefix`` (optional): folded into the prompt string —
    see :func:`t2i` for the rationale.
    """
    cfg = _config()
    url = f"{cfg['base_url']}/images/edits"
    files = {"image": _image_to_multipart(image)}
    final_prompt = _wrap_with_history(prompt, history_text_prefix)
    form = {
        "prompt": final_prompt,
        "size": _resolve_size(size, aspect_ratio, image_size),
        "output_format": output_format,
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    log.debug(
        "tools.it2i.request",
        url=url,
        size=form["size"],
        output_format=output_format,
        prompt_chars=len(final_prompt),
        has_history=bool(history_text_prefix),
        image_bytes=len(files["image"][1]),
    )
    started = time.monotonic()
    r = requests.post(
        url, files=files, data=form, headers=headers, timeout=DEFAULT_TIMEOUT
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if r.status_code >= 400:
        log.error(
            "tools.it2i.http_error",
            url=url,
            status=r.status_code,
            elapsed_ms=elapsed_ms,
            body_preview=r.text[:500],
        )
        raise RuntimeError(
            f"Images Edits API HTTP {r.status_code} from {url}: {r.text[:500]}"
        )
    try:
        data = r.json()
    except ValueError as e:
        log.error(
            "tools.it2i.non_json_response",
            url=url,
            elapsed_ms=elapsed_ms,
            body_preview=r.text[:500],
        )
        raise RuntimeError(
            f"Images Edits API non-JSON response: {r.text[:500]}"
        ) from e
    if isinstance(data, dict) and data.get("error"):
        log.error(
            "tools.it2i.api_error",
            url=url,
            elapsed_ms=elapsed_ms,
            error=data["error"],
        )
        raise RuntimeError(f"Images Edits API error: {data['error']}")
    image_bytes = _decode_b64_image(data)
    log.debug(
        "tools.it2i.response",
        url=url,
        status=r.status_code,
        elapsed_ms=elapsed_ms,
        image_bytes=len(image_bytes),
    )
    return image_bytes
