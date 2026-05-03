# The repo will show hands-on projects on NLP, Computer Vision, LLM

curl -X POST "https://bedrock-runtime.us-east-1.amazonaws.com/model/minimax.minimax-2.5/converse" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK" \
  -d '{
    "messages": [
        {
            "role": "user",
            "content": [{"text": "Hello"}]
        }
    ]
  }'


export ANTHROPIC_DEFAULT_OPUS_MODEL='minimax.minimax-m2.5'
export ANTHROPIC_DEFAULT_SONNET_MODEL='minimax.minimax-m2.5'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='minimax.minimax-m2.5'