import json
import uuid
import tiktoken
from typing import Dict, Any
import httpx
from constants import config

# --- Transformation Logic ---

def convert_image_source(source: Dict[str, Any]) -> str:
    """Converts Anthropic Base64 to OpenAI Data URI."""
    if source.get("type") == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        return f"data:{media_type};base64,{data}"
    return source.get("url", "")

def transform_request_body(anthropic_body: Dict[str, Any]) -> Dict[str, Any]:
    """Translates Anthropic v1/messages JSON to OpenAI v1/chat/completions JSON."""
    openai_messages = []
    
    # 1. System Prompt
    if "system" in anthropic_body:
        system_content = anthropic_body["system"]
        if isinstance(system_content, list):
            text_parts = [b["text"] for b in system_content if b["type"] == "text"]
            system_content = "\n".join(text_parts)
        openai_messages.append({"role": "system", "content": system_content})

    # 2. Message History
    for msg in anthropic_body.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            openai_content = []
            if isinstance(content, str):
                openai_content = content
            elif isinstance(content, list):
                # Check if this is a Tool Result block or Standard Content
                is_tool_result = any(b.get("type") == "tool_result" for b in content)
                
                if is_tool_result:
                    # Separate Tool Results into distinct messages
                    for block in content:
                        if block["type"] == "tool_result":
                            # OpenAI expects string content for tools
                            tool_content = ""
                            if isinstance(block.get("content"), str):
                                tool_content = block["content"]
                            elif isinstance(block.get("content"), list):
                                # Flatten mixed content in tool result
                                tool_content = " ".join([c.get("text", "") for c in block["content"] if c.get("type")=="text"])
                            
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": tool_content or "Success"
                            })
                    continue 
                else:
                    # Standard User Multimodal Message
                    for block in content:
                        if block["type"] == "text":
                            openai_content.append({"type": "text", "text": block["text"]})
                        elif block["type"] == "image":
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {"url": convert_image_source(block["source"])}
                            })
            
            openai_messages.append({"role": "user", "content": openai_content})

        elif role == "assistant":
            openai_msg = {"role": "assistant"}
            if isinstance(content, str):
                openai_msg["content"] = content
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if block["type"] == "text":
                        text_parts.append(block["text"])
                    elif block["type"] == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"])
                            }
                        })
                
                if text_parts:
                    openai_msg["content"] = "\n".join(text_parts)
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls
            
            openai_messages.append(openai_msg)

    # 3. Tools
    tools = []
    if "tools" in anthropic_body:
        for tool in anthropic_body["tools"]:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"]
                }
            })

    # 4. Construct Final Body
    openai_body = {
        "model": anthropic_body.get("model"),
        "messages": openai_messages,
        "stream": anthropic_body.get("stream", False),
        "max_tokens": anthropic_body.get("max_tokens", 4096),
        "temperature": anthropic_body.get("temperature", 0.7),
    }
    
    if openai_messages[-1]["role"]=="assistant":
        openai_body["continue_final_message"] = True
        openai_body["add_generation_prompt"] = False

    # Optional params
    if "stop_sequences" in anthropic_body:
        openai_body["stop"] = anthropic_body["stop_sequences"]
    if "top_p" in anthropic_body:
        openai_body["top_p"] = anthropic_body["top_p"]
    if "presence_penalty" in anthropic_body:
        openai_body["presence_penalty"] = anthropic_body["presence_penalty"]
    if "frequency_penalty" in anthropic_body:
        openai_body["frequency_penalty"] = anthropic_body["frequency_penalty"]

    if tools:
        openai_body["tools"] = tools
        if anthropic_body.get("tool_choice"):
            tc = anthropic_body["tool_choice"]
            if tc["type"] == "any": openai_body["tool_choice"] = "required"
            elif tc["type"] == "auto": openai_body["tool_choice"] = "auto"
            elif tc["type"] == "tool": 
                openai_body["tool_choice"] = {
                    "type": "function", 
                    "function": {"name": tc["name"]}
                }

    return openai_body

def transform_openai_response(openai_resp: Dict[str, Any]) -> Dict[str, Any]:
    """Converts OpenAI response to Anthropic Message format."""
    choice = openai_resp["choices"][0]
    message = choice["message"]
    
    content_blocks = []
    if message.get("content"):
        content_blocks.append({"type": "text", "text": message["content"]})
        
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            content_blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"])
            })

    finish_reason = choice.get("finish_reason")
    stop_reason = "end_turn"
    if finish_reason == "tool_calls": stop_reason = "tool_use"
    elif finish_reason == "length": stop_reason = "max_tokens"
    
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": openai_resp.get("model", "unknown"),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_resp.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_resp.get("usage", {}).get("completion_tokens", 0)
        }
    }

# --- Streaming Logic ---

async def stream_openai_to_anthropic(response: httpx.Response):
    msg_id = f"msg_{uuid.uuid4().hex}"
    
    # 1. Message Start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': 'proxy', 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    
    # 2. Start with Index 0 (Text Block)
    # Even if empty, Anthropic expects a block to be open or sequence of blocks.
    current_block_index = 0
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': current_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    async for line in response.aiter_lines():
        if not line.strip() or line.strip() == "data: [DONE]": 
            continue
        if line.startswith("data: "):
            try:
                chunk = json.loads(line[6:])
                
                if not chunk.get("choices"): continue
                
                delta = chunk["choices"][0]["delta"]
                
                # --- Text Content ---
                if delta.get("content") is not None:
                    # If we were previously handling a tool (index > 0), we must close it.
                    # But OpenAI usually sends Text -> Tools. Mixed order isn't standard.
                    # We assume text always belongs to block 0.
                    if current_block_index != 0:
                         # This edge case implies OpenAI switched back to text after tools.
                         # We'd need to close tool block and resume text block (unsupported by standard Anthropic flow usually)
                         # ignoring for now, assuming standard flow.
                         pass

                    if delta["content"]:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"

                # --- Tool Calls ---
                if delta.get("tool_calls"):
                    tc = delta["tool_calls"][0]
                    # OpenAI tools index 0, 1, 2...
                    # Anthropic Blocks: 0=Text, 1=Tool1, 2=Tool2...
                    target_index = tc["index"] + 1
                    
                    if target_index != current_block_index:
                        # Close the previous block (whether text or previous tool)
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"
                        
                        current_block_index = target_index
                        
                        # Start new Tool Use Block
                        # ID and Name are usually in the first chunk of a new tool index
                        tool_id = tc.get("id", "pending")
                        tool_name = tc.get("function", {}).get("name", "pending")
                        
                        start_payload = {
                            "type": "content_block_start",
                            "index": current_block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool_name,
                                "input": {}
                            }
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(start_payload)}\n\n"
                    
                    # Stream Arguments
                    if "function" in tc and "arguments" in tc["function"]:
                        args = tc["function"]["arguments"]
                        if args:
                             delta_payload = {
                                "type": "content_block_delta",
                                "index": current_block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args
                                }
                             }
                             yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n"

                # --- Stop Reason ---
                if chunk["choices"][0].get("finish_reason"):
                    reason = chunk["choices"][0]["finish_reason"]
                    anthropic_reason = "tool_use" if reason == "tool_calls" else "end_turn"
                    
                    # Close the currently open block
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"
                    
                    # Send Message Delta
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': anthropic_reason, 'stop_sequence': None}, 'usage': {'output_tokens': 10}})}\n\n"

            except Exception as e:
                print(f"Streaming Error: {e}")
                continue

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

# --- Token Counting Logic ---

def count_openai_tokens(openai_body: Dict[str, Any]) -> int:
    try:
        encoding = tiktoken.get_encoding(config.TIKTOKEN_ENCODING)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    messages = openai_body.get("messages", [])
    num_tokens = 0
    for message in messages:
        num_tokens += 3
        for key, value in message.items():
            if key == "content" and isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            elif key == "tool_calls":
                for tc in value:
                    num_tokens += len(encoding.encode(tc["function"]["name"]))
                    num_tokens += len(encoding.encode(tc["function"]["arguments"]))
    
    num_tokens += 3 
    if "tools" in openai_body:
        tools_str = json.dumps(openai_body["tools"])
        num_tokens += len(encoding.encode(tools_str))

    return num_tokens