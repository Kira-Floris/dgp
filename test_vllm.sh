curl http://localhost:4000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Kira-Floris/Qwen3-4B",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'