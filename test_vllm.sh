curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openai/gpt-oss-120b",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'