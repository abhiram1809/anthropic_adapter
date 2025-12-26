# cli.py
import argparse
from main import build_openai_api
from constants import config

def main():
    parser = argparse.ArgumentParser(description="Anthropic-to-OpenAI API Adapter")
    
    parser.add_argument("--base-url", type=str, default=config.OPENAI_BASE_URL, 
                        help="Target OpenAI-compatible API URL")
    parser.add_argument("--api-key", type=str, default=config.OPENAI_API_KEY, 
                        help="Target OpenAI API Key (can also pass via x-api-key header)")
    parser.add_argument("--host", type=str, default=config.HOST, 
                        help="Host to bind the adapter to")
    parser.add_argument("--port", type=int, default=config.PORT, 
                        help="Port to bind the adapter to")

    args = parser.parse_args()

    build_openai_api(
        base_url=args.base_url,
        api_key=args.api_key,
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main()