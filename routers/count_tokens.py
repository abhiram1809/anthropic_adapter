# routers/count_tokens.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from utils import transform_request_body, count_openai_tokens

router = APIRouter()

@router.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: Request):
    """
    Simulates Anthropic's count_tokens API.
    Since we proxy to OpenAI, we calculate tokens using OpenAI's tokenizer (tiktoken)
    on the transformed body.
    """
    try:
        anthropic_body = await request.json()
        
        # Transform to what we would send to OpenAI
        openai_body = transform_request_body(anthropic_body)
        
        # Calculate tokens
        token_count = count_openai_tokens(openai_body)
        
        return JSONResponse(content={"input_tokens": token_count})
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})