#!/bin/bash
# Reference t2t (text -> text) request — matches `app.tools.t2t`.
# Targets the dedicated text chat-completions endpoint (T2T_* env vars,
# default qwen3-30b-a3b on llama.cpp). `modalities: ["text"]` is kept as a
# safe fallback for callers who point this at a vllm-omni server instead.

set -euo pipefail

BASE_URL="${T2T_OPENAI_BASE_URL:-${OPENAI_API_BASE:-http://0.0.0.0:8889/v1}}"
MODEL_ID="${T2T_OPENAI_MODEL:-${OPENAI_MODEL_NAME:-qwen3-30b-a3b}}"
API_KEY="${T2T_OPENAI_API_KEY:-${OPENAI_API_KEY:-none}}"
USER_PROMPT="${1:-Say hi to the team in Chinese.}"

curl -sS -X POST "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "$(jq -n \
        --arg model "$MODEL_ID" \
        --arg prompt "$USER_PROMPT" \
        '{
          model: $model,
          messages: [
            {role: "system", content: "You are a helpful assistant."},
            {role: "user",   content: $prompt}
          ],
          modalities: ["text"],
          stream: false,
          temperature: 0.7,
          max_tokens: 4096,
          chat_template_kwargs: {enable_thinking: false}
        }')" \
  | jq -r '.choices[0].message.content'
