# main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from constants import config
from routers import messages, count_tokens

def create_app():
    app = FastAPI(title="Anthropic Adapter")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(messages.router)
    app.include_router(count_tokens.router)
    return app

def build_openai_api(
    base_url: str = None, 
    api_key: str = None, 
    host: str = "0.0.0.0", 
    port: int = 8000
):
    """
    Builder function to configure and run the server.
    """
    # Update singleton config
    config.update(base_url=base_url, api_key=api_key)
    
    app = create_app()
    print(f"🚀 Starting Anthropic Adapter on {host}:{port}")
    print(f"🔗 Proxying to: {config.OPENAI_BASE_URL}")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # Default entry point using env vars
    app = create_app()
    uvicorn.run(app, host=config.HOST, port=config.PORT)