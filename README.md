# minimal-agent

A tiny single-turn multimodal agent that bridges a local OpenAI-compatible LLM
endpoint and Feishu (Lark). It can:

- **Serve** as a long-connection chatbot (uses the official
  [`lark-oapi`](https://github.com/larksuite/oapi-sdk-python) SDK over
  WebSocket). The bot receives text / image messages from a Feishu user or
  group and replies in-thread with model output. **No public URL, HTTPS, or
  ngrok required.**
- **Run** a one-shot CLI dispatch and deliver the result into a specific chat
  via the IM Open API.

## 目标 / Goal

Provide four single-turn capabilities backed by the same OpenAI-compatible
endpoint:

| Tool   | 中文                          | English |
|--------|-------------------------------|---------|
| `t2t`  | 文字生成文字                  | text → text |
| `t2i`  | 文字生成图片                  | text → image |
| `it2t` | 文字 + 图片生成文字           | image + text → text (vision) |
| `it2i` | 图片 + 文字编辑生成图片       | image + text → image (edit) |

Each event is still a single tool call — there is no autonomous multi-step
agent loop — but when the bot is @-mentioned in a group (or DMed in P2P) it
will pull recent chat history from Feishu via `im.v1.message.list` and feed
it to the LLM as conversation context (see *Conversation history context*
below).

The HTTP payloads sent to the LLM are kept consistent with the reference shell
scripts in `requests_example/` and are covered by unit tests in `tests/`.

## Quick start

This project is managed with [uv](https://docs.astral.sh/uv/) (`brew install uv`
or see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
# 1. install deps (creates .venv/ and uv.lock automatically)
uv sync

# 2. configure (copy and fill in)
cp .env.example .env
$EDITOR .env   # set FEISHU_APP_ID + FEISHU_APP_SECRET (and OPENAI_BASE_URL if needed)

# 3a. run the bot — connects to Feishu over WebSocket and replies to messages
uv run minimal-agent serve
# … or use the bundled wrapper:
./app/start-lark.sh

# 3b. one-shot push into a chat (requires FEISHU_CHAT_ID or --chat-id)
uv run minimal-agent run "say hi to the team in Chinese"
uv run minimal-agent run "draw a cyberpunk cat" --mode t2i
uv run minimal-agent run "describe this image" -i ./input.jpg --mode it2t
uv run minimal-agent run "turn this into watercolor" -i ./input.jpg --mode it2i
uv run minimal-agent run "local only, no Feishu push" --no-feishu
```

`--mode` is the recommended way to pick a tool. Without it the agent uses a
deterministic keyword router (Chinese + English) for draw/edit intents. Set
`ENABLE_LLM_ROUTER=1` to opt into LLM-based tool selection instead — the
router uses OpenAI's standard `tools` / function-calling protocol against
the `T2T_*` endpoint and falls back to the keyword router on any
backend/parse failure.

### Managing dependencies with uv

```bash
uv add <package>           # add a runtime dep (updates pyproject.toml + uv.lock)
uv lock --upgrade          # refresh the lock file
uv sync                    # install exactly what uv.lock pins
uv run <cmd>               # run <cmd> inside the project venv
```

## One-time Feishu setup

The bot uses Feishu's *long-connection event subscription* — your app
out-connects to Feishu over WebSocket and receives events through that
connection. **No reverse proxy, ngrok or webhook URL needed.**

1. Open <https://open.feishu.cn/app> and click **Create Custom App** (自建应用).
2. In **Credentials & Basic Info** copy `App ID` and `App Secret` into
   `.env` as `FEISHU_APP_ID` / `FEISHU_APP_SECRET`.
3. In **Permissions & Scopes**, grant at minimum:
   - `im:message` (receive messages)
   - `im:message:send_as_bot` (send replies)
   - `im:resource` (download user-uploaded images)
   - `im:message:readonly` *(only required if `FEISHU_HISTORY_ENABLED=1`,
     which is the default — used by `im.v1.message.list` to load
     conversation history when @-mentioned or in P2P)*
4. In **Features → Bot**, enable the bot and add it to the chats you want to
   chat with.
5. In **Events & Callbacks → Event Subscription**:
   - Choose the **长连接 / Long Connection** delivery method (this is what
     `lark.ws.Client` consumes — no Request URL needed).
   - Subscribe to **`im.message.receive_v1`** (机器人接收用户消息).

Now `uv run minimal-agent serve` will connect and the bot will reply to
messages sent to it.

### How the bot decides whether to respond

`FEISHU_RESPOND_MODE` controls this (default `all`):

- `all` — reply to every event Feishu delivers. Note that **by default Feishu
  only delivers group messages to bots that are @-mentioned**, so this is the
  natural setting for most apps.
- `mentions_or_p2p` — defense-in-depth: in P2P chats always reply; in groups
  only reply if `FEISHU_BOT_OPEN_ID` is set and is mentioned in the message.

The bot also:

- **Filters out non-user senders** (`sender_type != "user"`) to avoid loops.
- **Dedups** events by `event_id` (LRU set) to defend against reconnect
  redelivery.
- **Offloads** LLM/image-generation work to a bounded `ThreadPoolExecutor` so
  the SDK's asyncio receive loop stays responsive.
- **Replies in-thread** via `POST /im/v1/messages/{message_id}/reply` so
  conversations look natural in groups, with `uuid = "{event_id}:{tool}"` for
  idempotency.
- **Downloads images** (including images embedded in `post` rich-text messages)
  via `client.im.v1.message_resource.get`.
- **Resolves quoted images** — if a user *quotes* (引用) an earlier image
  message and adds text like "改成黑白" / "make it watercolor", the bot fetches
  the parent message via `client.im.v1.message.get`, downloads its picture,
  and runs `it2i` against it. If the parent lookup fails (missing scope, etc.)
  it falls back to text-only handling; if the parent is found but the image
  itself can't be downloaded, the user gets a clear "无法读取被引用的图片" reply
  instead of silent failure.
- **Best-effort error replies** — if a worker raises, the bot tries to reply
  with a short error so the user isn't left waiting.

### Conversation history context

When the bot is engaged (P2P chat, or @-mentioned in a group) it pulls
recent messages from the same chat via `im.v1.message.list` and passes
them to the LLM as the OpenAI `messages` history. Concretely:

- **Scope** — only loads in P2P chats and `@`-mentioned group messages.
  In groups, mention detection requires `FEISHU_BOT_OPEN_ID`; without it
  the bot conservatively skips history loading and logs once at INFO.
- **Filter** — keeps only messages where the sender is the current user
  or the bot (`history_filter=self_and_bot`); side chatter from other
  group members is dropped.
- **Window** — most recent `FEISHU_HISTORY_COUNT` messages (default
  `20`, Feishu caps at 50) within the last `FEISHU_HISTORY_WINDOW_MINUTES`
  minutes (default `60`; set `0` for no time limit).
- **Images in history** — only included when `T2T_MULTIMODAL=1` (the
  T2T chat endpoint must accept `image_url` message parts). When unset,
  historical images are dropped and a single WARNING is emitted listing
  how many were skipped. The hard cap is `FEISHU_HISTORY_MAX_IMAGES`
  (default `3`), counted across history newest-first.
- **`t2i` / `it2i`** — the Images API has no multi-turn shape, so the
  history is folded into the prompt as a plaintext `Conversation
  context: … Current request: …` prefix.
- **Failure modes** — any failure loading history (missing scope,
  network blip, etc.) degrades silently to "no history" so the primary
  reply path is never broken.

Disable entirely with `FEISHU_HISTORY_ENABLED=0`.

## Configuration

All configuration is via environment variables (a `.env` file is auto-loaded
if present). See `.env.example`.

### Required
| Var | Purpose |
|---|---|
| `FEISHU_APP_ID` | Self-built app credentials. |
| `FEISHU_APP_SECRET` | Self-built app credentials. |

### Bot tuning (optional)
| Var | Default | Purpose |
|---|---|---|
| `FEISHU_RESPOND_MODE` | `all` | `all` or `mentions_or_p2p`. |
| `FEISHU_BOT_OPEN_ID` | — | Required for `mentions_or_p2p` *and* for reliable history loading in groups (used to detect bot @-mentions). |
| `FEISHU_WORKER_THREADS` | `4` | Worker pool size for LLM calls. |
| `FEISHU_DEDUP_CAPACITY` | `1024` | LRU size for `event_id` dedup. |
| `LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `LOG_FORMAT` | `auto` | `console` (human-readable, colored if TTY), `json` (one-line, log-aggregator friendly), or `auto` (console on TTY, JSON otherwise). |
| `FEISHU_DEBUG` | — | Legacy alias. Set to `1` to force `LOG_LEVEL=DEBUG` *and* enable verbose lark_oapi SDK logs. |

### Conversation history (optional)
| Var | Default | Purpose |
|---|---|---|
| `FEISHU_HISTORY_ENABLED` | `1` | Kill switch. Set to `0` to disable history loading. |
| `FEISHU_HISTORY_COUNT` | `20` | Max recent messages to include (Feishu caps at 50). |
| `FEISHU_HISTORY_WINDOW_MINUTES` | `60` | Only include messages from the last N minutes. `0` = unlimited. |
| `FEISHU_HISTORY_MAX_IMAGES` | `3` | Hard cap on history images included in the LLM call (newest-first). Ignored when `T2T_MULTIMODAL` is not set. |
| `T2T_MULTIMODAL` | `0` | Set to `1` if the T2T endpoint accepts `image_url` parts. Required to feed historical images to the LLM. When unset, history images are dropped and a WARNING is logged. |

### `run` mode
| Var | Purpose |
|---|---|
| `FEISHU_CHAT_ID` | Target chat (`oc_xxxx…`). Override per-run with `--chat-id`. |

### LLM endpoint
| Var | Default | Purpose |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible base URL. |
| `OPENAI_API_KEY`  | `dummy` | Bearer auth. |
| `OPENAI_MODEL`    | `model` | Model id passed to the endpoint. |

### Misc
| Var | Default | Purpose |
|---|---|---|
| `OUTPUT_DIR` | `./output_images` | Where generated images are saved. |
| `ENABLE_LLM_ROUTER` | `0` | Set to `1` to let the LLM choose the tool via OpenAI tools/function-calling (requires a backend that supports `tools`; falls back to the keyword router on miss). |
| `LLM_TIMEOUT` | `300` | Per-LLM-request timeout in seconds. |

## Project layout

```
minial-agent/
├── app/
│   ├── __main__.py    # CLI entry  (`minimal-agent run | serve`)
│   ├── agent.py       # routing + dispatch (run + handle_feishu_event)
│   ├── tools.py       # the four LLM capability functions
│   ├── feishu.py      # thin lark-oapi wrappers (send/reply/upload/download/list)
│   ├── history.py     # builds OpenAI chat history from Feishu messages
│   ├── _messages.py   # shared Feishu message-content parsers
│   ├── bot.py         # long-connection bot (lark.ws.Client + worker pool)
│   ├── start-lark.sh  # convenience wrapper around `uv run minimal-agent serve`
│   └── lark-samples-main/  # reference samples from Feishu (read-only)
├── tests/             # offline unit tests (no network access required)
├── requests_example/  # reference shell scripts; source of truth for LLM payload shape
├── pyproject.toml     # uv source of truth
├── uv.lock
├── .env.example
└── README.md
```

## Testing

```bash
uv run python -m unittest discover -v
```

All tests run offline (HTTP and SDK calls are monkeypatched). Coverage:

- LLM payload shapes match the reference scripts in `requests_example/`.
- The keyword and (optional) LLM tool router.
- Feishu helpers exercise the SDK builder/request flow with a stubbed client.
- The bot handler: message parsing, image download dispatch, dedup,
  sender_type filter, mention-gating, and the error-reply path.

## Exit codes (`run` mode)

| Code | Meaning |
|---|---|
| `0` | Success (and Feishu delivery succeeded, unless `--no-feishu`). |
| `1` | Generic error (e.g. LLM call failed). |
| `2` | Bad invocation (missing image, bad mode, …). |
| `3` | Local generation succeeded but Feishu delivery failed. |
