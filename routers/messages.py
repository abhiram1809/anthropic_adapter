# routers/messages.py
import httpx
from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from constants import config
from utils import transform_request_body, transform_openai_response, stream_openai_to_anthropic

router = APIRouter()

@router.post("/v1/messages")
async def create_message(request: Request, x_api_key: str = Header(None)):
    try:
        # 1. Configuration Validation
        target_api_key = x_api_key or config.OPENAI_API_KEY
        if not target_api_key:
            raise HTTPException(status_code=401, detail="Missing API Key. Provide via x-api-key header or .env")

        # 2. Transform Request
        anthropic_body = await request.json()
        openai_body = transform_request_body(anthropic_body)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {target_api_key}"
        }

        # 3. Forward Request
        client = httpx.AsyncClient()
        req = client.build_request(
            "POST",
            config.OPENAI_BASE_URL,
            json=openai_body,
            headers=headers,
            timeout=60.0
        )

        # 4. Handle Response
        if openai_body.get("stream"):
            print("Streaming is true")
            response = await client.send(req, stream=True)
            return StreamingResponse(
                stream_openai_to_anthropic(response),
                media_type="text/event-stream"
            )
        else:
            response = await client.send(req)
            if response.status_code != 200:
                # Transform OpenAI error to Anthropic format
                error_resp = response.json()
                anthropic_error = {
                    "type": "error",
                    "error": {
                        "type": error_resp.get("error", {}).get("type", "invalid_request_error"),
                        "message": error_resp.get("error", {}).get("message", "Unknown error")
                    }
                }
                return JSONResponse(status_code=response.status_code, content=anthropic_error)

            anthropic_resp = transform_openai_response(response.json())
            print("Response Recieved !!!")
            print(anthropic_resp)
            return JSONResponse(content=anthropic_resp)

    except Exception as e:
        print(f"Error: {e}")
        # Return proper Anthropic error format
        error_response = {
            "type": "error",
            "error": {
                "type": "internal_server_error",
                "message": str(e)
            }
        }
        raise HTTPException(status_code=500, detail=error_response)