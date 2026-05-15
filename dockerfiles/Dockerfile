# syntax=docker/dockerfile:1.7
#
# Build a self-contained `minimal-agent` image for linux/amd64.
#
# Build from the repo root so the context contains pyproject.toml, uv.lock,
# and app/:
#
#   docker buildx build \
#       --platform linux/amd64 \
#       -t minimal-agent:latest \
#       -f dockerfiles/Dockerfile .
#
# … or use `docker compose -f dockerfiles/docker-compose.yml build`.
#
# The image follows the project rule (AGENTS.md) of managing Python
# dependencies exclusively through uv — never pip / poetry — and installs
# the project non-editable so the runtime stage can drop uv entirely.

ARG PYTHON_VERSION=3.12

# ---------- builder ----------
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim-bookworm AS builder

# Pull a pinned uv binary out of the official image — keeps the lockfile
# format compatible with the uv version that produced uv.lock locally.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Install third-party deps first so they layer-cache independently of the
# project source. README.md is referenced from pyproject.toml.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Now install the project itself on top of the cached deps. `--no-editable`
# materialises app/ into the venv so the runtime stage can stand alone
# without re-copying the source tree.
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------- runtime ----------
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    OUTPUT_DIR=/app/output_images

WORKDIR /app

# Pull in the already-populated virtualenv (deps + the project + console
# script entry point at /opt/venv/bin/minimal-agent).
COPY --from=builder /opt/venv /opt/venv

# Pre-create writable dirs the bot uses at runtime and drop privileges.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin agent \
    && mkdir -p /app/output_images /app/inputs \
    && chown -R agent:agent /app
USER agent

# Sanity-check the install at build time.
RUN minimal-agent --help > /dev/null

# Default to `serve`; override with e.g. `docker run … run "prompt"`.
ENTRYPOINT ["minimal-agent"]
CMD ["serve"]
