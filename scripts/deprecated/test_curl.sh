curl http://localhost:96886/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 64,
      "temperature": 0
    }
  }'
