# routers/messages.py
import httpx, json
from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from constants import config
from utils import (
    transform_request_body,
    transform_openai_response,
    stream_openai_to_anthropic,
    transform_request_body_v1_responses,
    transform_v1_responses_response,
    stream_v1_responses_to_anthropic
)

router = APIRouter()

@router.post("/v1/messages")
async def create_message(request: Request, x_api_key: str = Header(None)):
    try:
        # 1. Configuration Validation
        target_api_key = x_api_key or config.OPENAI_API_KEY
        if not target_api_key:
            raise HTTPException(status_code=401, detail="Missing API Key. Provide via x-api-key header or .env")

        # 2. Transform Request based on API type
        anthropic_body = await request.json()
        print(json.dumps(anthropic_body, indent=4))

        # Determine which transformation to use based on API type
        if config.API_TYPE == "v1_responses":
            print(f"Using v1/responses API transformation")
            target_body = transform_request_body_v1_responses(anthropic_body)
            transform_response_fn = transform_v1_responses_response
            stream_fn = stream_v1_responses_to_anthropic
        else:
            print(f"Using v1/chat/completions API transformation")
            target_body = transform_request_body(anthropic_body)
            transform_response_fn = transform_openai_response
            stream_fn = stream_openai_to_anthropic

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {target_api_key}"
        }

        # 3. Forward Request
        client = httpx.AsyncClient()
        req = client.build_request(
            "POST",
            config.OPENAI_BASE_URL,
            json=target_body,
            headers=headers,
            timeout=60.0
        )

        # 4. Handle Response
        if target_body.get("stream"):
            print("Streaming is true")
            response = await client.send(req, stream=True)
            return StreamingResponse(
                stream_fn(response),
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

            anthropic_resp = transform_response_fn(response.json())
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