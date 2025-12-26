from main import build_openai_api

# This will block and run the server
build_openai_api(
    base_url="https://api.groq.com/openai/v1/chat/completions", # Example: Groq
    api_key="gsk_...",
    port=8080
)

# Run via CLI

# ```bash
# python cli.py --port 9000 --base-url "http://localhost:1234/v1/chat/completions"
# ```

# Or via Anthropic SDK

# from anthropic import Anthropic

# client = Anthropic(
#     base_url="http://localhost:8080/v1",
#     api_key="dummy" # The adapter maps this to Bearer
# )

# # 1. Token Counting
# count = client.messages.count_tokens(
#     model="gpt-4",
#     messages=[{"role": "user", "content": "Hello world"}]
# )
# print(f"Tokens: {count.input_tokens}")

# # 2. Chat
# msg = client.messages.create(
#     model="gpt-4",
#     max_tokens=100,
#     messages=[{"role": "user", "content": "Hello!"}]
# )
# print(msg.content[0].text)