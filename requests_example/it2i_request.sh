#!/bin/bash
# Reference it2i request — matches `app.tools.it2i` (and it2i_example.sh).
# Uses the OpenAI Images Edits API (multipart) on the multimodal endpoint.

set -euo pipefail

BASE_URL="${OPENAI_BASE_URL:-http://0.0.0.0:8000/v1}"
INPUT_IMAGE="${1:-./input_images/input.jpg}"
USER_PROMPT="${2:-Turn this image into a cyberpunk style, neon lights, 8k}"
OUT="${3:-./output_images/$(date +%Y%m%d_%H%M%S)_it2i.png}"

if [ ! -f "$INPUT_IMAGE" ]; then
  echo "[ERROR] input image not found: $INPUT_IMAGE"
  echo "usage: $0 [input_image] [prompt] [output_path]"
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

curl -sS -X POST "${BASE_URL}/images/edits" \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-dummy}" \
  -F "image=@${INPUT_IMAGE}" \
  -F "prompt=${USER_PROMPT}" \
  -F "size=1024x1024" \
  -F "output_format=png" \
  | jq -r '.data[0].b64_json' \
  | base64 --decode > "$OUT"

echo "[SUCCESS] saved $OUT ($(ls -lh "$OUT" | awk '{print $5}'))"
