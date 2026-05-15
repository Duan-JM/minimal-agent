# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

`minimal-agent` is a tiny single-turn multimodal agent that bridges a local
OpenAI-compatible LLM endpoint and Feishu (Lark). It provides:

- A **long-connection Feishu bot** (`lark-oapi` over WebSocket) that receives
  text / image messages from a Feishu user or group and replies in-thread —
  **no public URL, HTTPS, or ngrok required**.
- A **one-shot CLI** (`minimal-agent run …`) that dispatches a single prompt
  and optionally pushes the result into a chat via the IM Open API.
- Four single-turn capabilities backed by the same OpenAI-compatible endpoint:
  `t2t` (text→text), `t2i` (text→image), `it2t` (image+text→text, VQA),
  `it2i` (image+text→image, edit).
- A deterministic Chinese+English keyword tool router (with optional
  LLM-based routing behind `ENABLE_LLM_ROUTER=1`).

## Commands

This project is managed with [uv](https://docs.astral.sh/uv/) — **never use
`pip` or `poetry` directly**.

```bash
# Install / sync dependencies (creates .venv/ from uv.lock)
uv sync

# Run the CLI
uv run minimal-agent run "say hi to the team in Chinese"
uv run minimal-agent run "draw a cyberpunk cat" --mode t2i
uv run minimal-agent run "describe this image" -i ./input.jpg --mode it2t
uv run minimal-agent run "turn this into watercolor" -i ./input.jpg --mode it2i
uv run minimal-agent run "local only, no Feishu push" --no-feishu

# Start the long-connection Feishu bot
uv run minimal-agent serve
# … or use the bundled wrapper:
./app/start-lark.sh

# Run all tests (offline; HTTP and SDK calls are monkeypatched)
uv run python -m unittest discover -v

# Run a single test file
uv run python -m unittest tests.test_shapes -v
uv run python -m unittest tests.test_bot -v

# Managing dependencies
uv add <package>           # add a runtime dep (updates pyproject.toml + uv.lock)
uv lock --upgrade          # refresh the lock file
uv run <cmd>               # run <cmd> inside the project venv
```

No formatter, linter, type-checker, or coverage gate is currently
configured. Do **not** introduce one as a side effect of an unrelated change —
open a separate PR if tooling needs to be added.

## Architecture

### Source layout: `app/`

The whole package lives under `app/` (declared as the wheel target in
`pyproject.toml`). There is no `src/` layout.

**Entry point**: `app/__main__.py` — argparse-based CLI exposing two
subcommands, `run` (one-shot dispatch) and `serve` (long-connection bot).
Wired up via `[project.scripts] minimal-agent = "app.__main__:main"` and a
back-compat shim that rewrites `minimal-agent "prompt"` to
`minimal-agent run "prompt"`.

**Main modules:**

1. **`app/agent.py`** — routing + dispatch.
   - `run(prompt, image_path, mode, output_dir, notify_feishu, chat_id)` —
     the one-shot dispatch used by the `run` CLI; returns a `ToolResult`.
   - `handle_feishu_event(...)` — called by the bot worker pool for each
     deduped event.
   - `VALID_TOOLS`, `IMAGE_GEN_KEYWORDS`, `IMAGE_EDIT_KEYWORDS`,
     `DEFAULT_IMAGE_ONLY_PROMPT` — the keyword router’s source of truth.

2. **`app/tools.py`** — the four LLM capability functions, split across two
   endpoints:
   - Text-only chat endpoint (`T2T_*` env vars): `t2t`, `it2t` (VQA) via
     `POST /v1/chat/completions`.
   - Multimodal images endpoint (`OPENAI_*` env vars): `t2i` via
     `POST /v1/images/generations`, `it2i` via `POST /v1/images/edits`
     (multipart).
   - `_extract_text` strips the model’s leading `<think>…</think>` block
     before returning user-facing text.

3. **`app/feishu.py`** — thin wrappers over `lark.Client.im.v1`:
   `reply_text` / `reply_image`, `send_text` / `send_image`,
   `download_message_resource`, `upload_image`. Raises `FeishuError` on
   non-success API responses. The SDK handles tenant-token caching, signing,
   and retry — never touch raw HTTP from here.

4. **`app/bot.py`** — long-connection bot. Wires up `lark.ws.Client`, dedups
   events by `event_id` (LRU), offloads slow LLM/image work to a bounded
   `ThreadPoolExecutor`, and replies in-thread with `uuid = "{event_id}:{tool}"`
   for idempotency. Also resolves quoted-image replies by fetching the parent
   message and downloading its picture.

**Other top-level files:**

- `app/start-lark.sh` — convenience wrapper around `uv run minimal-agent serve`.
- `app/lark-samples-main/` — read-only reference samples from Feishu; do
  **not** modify.
- `requests_example/` — reference shell scripts that are the source of truth
  for LLM payload shape; the unit tests assert payloads match these.
- `official_example.py`, `t2i_example.sh`, `inputs/`, `output_images/` —
  reference / scratch material.

### Key patterns

- **Configuration**: All settings via environment variables. A `.env` file is
  auto-loaded by `python-dotenv` in `main()`. See `.env.example` and the
  Configuration table in `README.md` for the full list — required:
  `FEISHU_APP_ID`, `FEISHU_APP_SECRET`. Common: `OPENAI_BASE_URL`,
  `OPENAI_API_KEY`, `OPENAI_MODEL`, `T2T_*`, `FEISHU_CHAT_ID`,
  `FEISHU_RESPOND_MODE`, `FEISHU_BOT_OPEN_ID`, `FEISHU_WORKER_THREADS`,
  `FEISHU_DEDUP_CAPACITY`, `OUTPUT_DIR`, `ENABLE_LLM_ROUTER`, `LLM_TIMEOUT`.
- **No persistent storage**: no database, no cache, no broker. State is
  in-process (LRU dedup set, thread pool).
- **Service architecture**:
  - `run` mode: CLI → `agent.run` → `tools.<t2t|t2i|it2t|it2i>` → optional
    `feishu.send_*`.
  - `serve` mode: `lark.ws.Client` event → `bot._handle` → dedup +
    sender/mention gating → `ThreadPoolExecutor` → `agent.handle_feishu_event`
    → `tools.*` → `feishu.reply_*`.
- **Background / async work**: the SDK’s asyncio receive loop must stay
  responsive, so all LLM/image work is offloaded to a bounded
  `ThreadPoolExecutor` (`FEISHU_WORKER_THREADS`, default 4).
- **Logging**: standard library + `lark_oapi`’s own logger. Set
  `FEISHU_DEBUG=1` for verbose SDK logs. Do not introduce a new logging
  framework as a side effect.
- **Error handling**: `tools.py` raises `RuntimeError` / `ValueError` on bad
  responses; `feishu.py` raises `FeishuError`. The bot handler wraps worker
  exceptions and sends a best-effort short error reply so users aren’t left
  hanging. Exit codes for `run` mode: `0` ok, `1` generic error, `2` bad
  invocation, `3` local generation ok but Feishu delivery failed.
- **No legacy code**: deprecated or replaced code MUST be deleted, not left
  behind "for reference". When a module is superseded, remove the old files
  entirely and update all imports, tests, and documentation. (The
  `app/lark-samples-main/` directory is the one exception — it is upstream
  reference material.)

### Infrastructure

- **Docker Compose** (`docker-compose.yml`, at repo root): runs
  `vllm/vllm-omni` serving the multimodal images model (default
  `sensenova/SenseNova-U1-8B-MoT`) on `localhost:8000`. This is the
  backing endpoint for `t2i` / `it2i` during local development. The
  text-only `T2T_*` endpoint is **not** managed by this compose file —
  point it at a separate llama.cpp / vLLM instance.
- **Agent container** (`dockerfiles/Dockerfile`): multi-stage,
  `linux/amd64`-only build of `minimal-agent` itself, driven by `uv` (no
  pip/poetry). The builder installs the project + deps into `/opt/venv`
  with `uv sync --frozen --no-dev --no-editable`; the runtime stage
  drops `uv`, copies only the venv, and runs as a non-root `agent` user.
  Default `CMD` is `serve`; override with `run "…"` to invoke the
  one-shot CLI. Build from the repo root with
  `docker buildx build --platform linux/amd64 -t minimal-agent:latest -f dockerfiles/Dockerfile .`.
- **Agent deploy compose** (`dockerfiles/docker-compose.yml`):
  orchestrates the `minimal-agent serve` container. Reads `../.env`
  (repo-root `.env`), persists generated images via a bind mount on
  `../output_images`, and maps `host.docker.internal` to the host
  gateway so the container can reach a vllm backend running on the host.
  No ports are published — the bot is outbound-only (Feishu WebSocket).
  Bring up with `docker compose -f dockerfiles/docker-compose.yml up -d --build`.
- **`.dockerignore`** (repo root) trims the build context — keeps `.git`,
  `.venv`, `output_images`, `inputs`, `tests`, `requests_example`, and
  the upstream `app/lark-samples-main/` reference tree out of the image.
- No migration / dev scripts beyond `app/start-lark.sh`.

## Code Style

- **Formatter / linter / type-checker**: none configured. If you add type
  hints or docstrings, match the existing style in `app/`: PEP 8, 4-space
  indentation, type hints on public functions, module-level docstrings,
  `from __future__ import annotations` at the top of new modules.
- Prefer small focused functions over classes when there is no state — the
  existing modules (`tools.py`, `feishu.py`, `agent.py`) are function-first.
  Do not refactor them to classes without a clear reason.
- Keep public surface minimal: `agent.py` exports `run` and
  `handle_feishu_event`; `tools.py` exports `t2t`, `t2i`, `it2t`, `it2i`;
  `feishu.py` exports the small set listed in its module docstring. New
  helpers should be private (`_leading_underscore`) unless callers outside
  the module need them.
- Descriptive names; comments for non-obvious logic only.

## Testing

- Tests live in `tests/` at the repo root and use **`unittest`**, not pytest.
- Run with `uv run python -m unittest discover -v`.
- All tests run **offline**: HTTP (`requests`) and the `lark-oapi` SDK are
  monkeypatched. Do not add tests that hit the network or require Feishu /
  LLM credentials.
- Existing coverage:
  - LLM payload shapes match the reference scripts in `requests_example/`
    (`tests/test_shapes.py`).
  - Keyword router and (optional) LLM tool router.
  - `feishu.py` helpers exercise the SDK builder/request flow with a stubbed
    client.
  - Bot handler: message parsing, image download dispatch, dedup,
    `sender_type` filter, mention-gating, and the error-reply path
    (`tests/test_bot.py`).
- No pytest markers, no coverage gate. `tests/test_shapes.py` injects the
  repo root into `sys.path`, so imports are `from app.<module> import …`.

## Dependencies

- **uv** for all dependency management — **never** use `pip` or `poetry`
  directly. `pyproject.toml` + `uv.lock` are the source of truth.
- Python `>=3.9` (per `pyproject.toml`).
- Runtime deps (kept intentionally minimal):
  - `requests>=2.31` — HTTP calls to the OpenAI-compatible endpoints.
  - `python-dotenv>=1.0` — loads `.env` for both CLI subcommands.
  - `lark-oapi>=1.4.8` — official Feishu SDK; provides both the IM REST
    client and the long-connection `lark.ws.Client`.
- No dev dependency group is populated (`[dependency-groups] dev = []`).
  `unittest` is in the stdlib, so the test suite needs no extra deps.

## Iteration Workflow (MANDATORY for AI agents)

Every code change — feature, fix, refactor, docs, even one-line typos —
must go through this loop. **Direct pushes to `main` are forbidden**, no
exceptions. The loop ensures CI is the single source of truth for "is this
change safe to merge".

### The 6-step loop

1. **Branch from latest `main`**

   ```bash
   git checkout main && git pull --ff-only origin main
   git checkout -b <type>/<slug>
   ```

   `<type>` ∈ {`feat`, `fix`, `docs`, `refactor`, `test`, `chore`} —
   matches Conventional Commits.
   `<slug>` is 2–5 word kebab-case (e.g. `fix/login-redirect-loop`,
   `feat/csv-export`).

2. **Implement and verify locally** before pushing:

   ```bash
   uv sync
   uv run python -m unittest discover -v
   ```

   If you touched `pyproject.toml`, also run `uv lock` and commit the
   updated `uv.lock`. No formatter/linter/type-checker is configured — do
   not invent CI steps that don't exist locally.

3. **Commit** with Conventional Commits format. Every commit message must
   include the trailer:

   ```
   Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
   ```

4. **Push the branch and open a PR**:

   ```bash
   git push -u origin HEAD
   gh pr create --fill --base main
   ```

   The PR body must include a `## Verification` section listing exactly
   what was run locally (the commands from step 2 plus their outcomes).

5. **Watch CI and self-heal until green**:

   ```bash
   gh run watch --exit-status        # blocks until the run finishes
   # if it fails:
   gh run view <run-id> --log-failed # diagnose
   # push fix commits to the same branch, repeat
   ```

   **Hard limit: 3 fix attempts.** If CI is still red after the third
   push, stop. Summarize what was tried and surface the failure to the
   human — do NOT keep guessing. Suspected-flaky failures count toward
   this budget; if you believe a failure is flaky, say so explicitly in
   the PR and stop.

6. **Stop after the PR is green. Do NOT auto-merge.** Report the PR URL
   and the final green CI run ID. Merging is the human's call.

### Why no direct pushes to `main`

Changes that "look clean locally" can still fail on CI's cold environment.
The PR + CI loop catches those before they land on `main`, and gives
reviewers a single artifact (the PR diff) to inspect rather than a moving
`main`.
