#!/bin/bash
# Quick-start wrapper for the bot. Reads APP_ID / APP_SECRET from env or .env.
# For day-to-day use prefer `uv run minimal-agent serve` directly.
set -e
cd "$(dirname "$0")/.."
exec uv run minimal-agent serve
