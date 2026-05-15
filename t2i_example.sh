curl -X POST http://10.170.138.159:8891/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dragon laying over the spine of the Green Mountains of Vermont",
    "size": "1024x1024",
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > dragon.png
