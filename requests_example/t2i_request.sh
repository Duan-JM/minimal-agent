#!/bin/bash
# Reference t2i request — matches `app.tools.t2i` (and t2i_example.sh).
# Uses the OpenAI Images API on the multimodal endpoint (vllm/vllm-omni).

set -euo pipefail

BASE_URL="${OPENAI_BASE_URL:-http://0.0.0.0:8000/v1}"
USER_PROMPT="${1:-A cyberpunk cat sitting on a neon roof, 8k}"
OUT="${2:-./output_images/$(date +%Y%m%d_%H%M%S)_t2i.png}"

mkdir -p "$(dirname "$OUT")"

curl -sS -X POST "${BASE_URL}/images/generations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-dummy}" \
  -d "$(jq -n \
        --arg prompt "$USER_PROMPT" \
        '{prompt: $prompt, size: "1024x1024", seed: 42}')" \
  | jq -r '.data[0].b64_json' \
  | base64 --decode > "$OUT"

echo "[SUCCESS] saved $OUT ($(ls -lh "$OUT" | awk '{print $5}'))"
