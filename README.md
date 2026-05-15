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

Single-turn only — no conversation history.

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
| `FEISHU_BOT_OPEN_ID` | — | Required only if `FEISHU_RESPOND_MODE=mentions_or_p2p`. |
| `FEISHU_WORKER_THREADS` | `4` | Worker pool size for LLM calls. |
| `FEISHU_DEDUP_CAPACITY` | `1024` | LRU size for `event_id` dedup. |
| `FEISHU_DEBUG` | — | Set to `1` for verbose SDK logs. |

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
│   ├── feishu.py      # thin lark-oapi wrappers (send/reply/upload/download)
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
