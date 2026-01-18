# routers/count_tokens.py
# This module provides an API endpoint to count tokens in messages
# following the Anthropic API specification, but using OpenAI's tokenizer

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from utils import transform_request_body, count_openai_tokens

# Create a FastAPI router instance for this endpoint
router = APIRouter()

@router.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: Request):
    """
    Simulates Anthropic's count_tokens API.

    This endpoint accepts a request body in Anthropic's message format,
    transforms it to OpenAI's format, and returns the token count using
    OpenAI's tiktoken tokenizer.

    Args:
        request: FastAPI Request object containing the JSON payload

    Returns:
        JSONResponse: A response containing the token count or error message

    Example request body:
        {
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ]
        }

    Example response:
        {"input_tokens": 15}
    """
    try:
        # Parse the incoming JSON request body
        # This contains the messages in Anthropic's format
        anthropic_body = await request.json()

        # Transform the Anthropic-formatted body to OpenAI's expected format
        # This conversion is necessary because we proxy to OpenAI's API
        # The transform_request_body function handles the conversion logic
        openai_body = transform_request_body(anthropic_body)

        # Calculate the token count using OpenAI's tokenizer
        # The count_openai_tokens function uses tiktoken to count tokens
        # in the OpenAI-formatted message body
        token_count = count_openai_tokens(openai_body)

        # Return the token count in the expected response format
        # The response follows Anthropic's API specification
        return JSONResponse(content={"input_tokens": token_count})

    except Exception as e:
        # If any error occurs during processing, return a 500 error
        # with the error message as the response body
        return JSONResponse(status_code=500, content={"error": str(e)})