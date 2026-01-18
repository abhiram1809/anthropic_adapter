# 🚀 Anthropic Adapter

![Anthropic Adapter Logo](Anthropic-Adapter.png "Anthropic Adapter")

An open-source API adapter that translates between Anthropic and OpenAI message formats, enabling seamless interoperability.

## 📋 Overview

The Anthropic Adapter is a lightweight FastAPI proxy server that bridges the gap between **Anthropic API** (v1/messages) and **OpenAI API** (v1/chat/completions and v1/responses). It transforms requests and responses between these two formats, allowing you to use any OpenAI-compatible LLM provider with Anthropic SDKs.

## 🎯 Features

- ✅ **Format Translation**: Automatically convert Anthropic → OpenAI and OpenAI → Anthropic message formats
- ✅ **Dual API Support**: Works with both `v1/chat/completions` and `v1/responses` endpoints
- ✅ **Auto-Detection**: Automatically detects and uses the appropriate API based on your base URL
- ✅ **Streaming Support**: Real-time SSE streaming with proper event formatting
- ✅ **Token Counting**: Built-in token counter compatible with OpenAI's tiktoken
- ✅ **Multimodal Support**: Handle text, images, and tool calls
- ✅ **Configurable**: Runtime configuration via environment variables or direct API
- ✅ **Error Handling**: Proper error transformation and reporting
- ✅ **CORS Support**: Configured to work with any frontend application

## 📦 Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/anthropic-adapter.git
   cd anthropic-adapter
   ```

2. Create a virtual environment:
   ```bash
   uv venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## 🚀 Usage

### Quick Start

Run the adapter with default configuration:

```bash
python run_adapter.py
```

Or with custom parameters:

```bash
python cli.py --port 9000 --base-url "http://localhost:1234/v1/chat/completions"
```

### Environment Variables

Create a `.env` file in the root directory:

```bash
# For v1/chat/completions (default)
OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions

# For v1/responses (newer API)
# OPENAI_BASE_URL=https://api.openai.com/v1/responses

OPENAI_API_KEY=your-openai-api-key-here
HOST=0.0.0.0
PORT=8000
TIKTOKEN_ENCODING=cl100k_base
```

**Note**: The adapter automatically detects which API format to use based on your `OPENAI_BASE_URL`. Simply change the URL to switch between `v1/chat/completions` and `v1/responses`.

### Programmatic Usage

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8080/v1",
    api_key="dummy"  # The adapter maps this to Bearer
)

# 1. Token Counting
count = client.messages.count_tokens(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello world"}]
)
print(f"Tokens: {count.input_tokens}")

# 2. Chat
msg = client.messages.create(
    model="gpt-4",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(msg.content[0].text)
```

### Using with Claude Code

The Anthropic Adapter can be used with [Claude Code](https://code.claude.com/docs/en/overview) to access various OpenAI-compatible models through the Anthropic SDK interface.

#### Configuration

To use the adapter with Claude Code, set the following environment variables:

```bash
# Set these in your shell (e.g., ~/.bashrc, ~/.zshrc)
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_AUTH_TOKEN="your-openai-api-key-here"
export ANTHROPIC_API_KEY=""  # Important: Must be explicitly empty
```

Or use a project-level settings file at `.claude/settings.local.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_AUTH_TOKEN": "your-openai-api-key-here",
    "ANTHROPIC_API_KEY": ""
  }
}
```

#### Model Configuration

You can configure Claude Code to use any model on your OpenAI-compatible API by overriding the default model aliases using environment variables:

```bash
export ANTHROPIC_DEFAULT_SONNET_MODEL="<MODEL_1>"
export ANTHROPIC_DEFAULT_OPUS_MODEL="<MODEL_2>"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="<MODEL_3>"
```

Replace `<MODEL_1>`, `<MODEL_2>`, and `<MODEL_3>` with the exact model names from your OpenAI-compatible API.

#### Usage

Once configured, you can use Claude Code to interact with any OpenAI-compatible API through the Anthropic interface:

```bash
cd /path/to/your/project
claude
```

Now any Anthropic SDK calls will be routed through the adapter to your configured OpenAI-compatible API endpoint.

## 🔧 Architecture

### Components

1. **main.py** - FastAPI application setup with CORS middleware
2. **constants.py** - Configuration management with environment variables
3. **utils.py** - Core transformation logic:
   - `transform_request_body()` - Anthropic → OpenAI conversion
   - `transform_openai_response()` - OpenAI → Anthropic conversion
   - `stream_openai_to_anthropic()` - Streaming SSE support
   - `count_openai_tokens()` - Token counting
4. **routers/messages.py** - API endpoint handlers
5. **routers/count_tokens.py** - Token counting endpoint

### Data Flow

```
Anthropic Format → [Transform] → OpenAI Format → [Forward] → OpenAI API → [Response] → [Transform] → Anthropic Format
```

## 🛠️ API Endpoints

### `/v1/messages`

**POST** endpoint for message creation and streaming.

**Request Headers:**
- `x-api-key`: Your OpenAI API key (or via environment variable)

**Request Body:**
- Standard Anthropic v1/messages JSON format
- Supports streaming with `stream: true`

**Response:**
- Anthroipc format message response
- Streaming uses Server-Sent Events (SSE) format

### `/v1/count_tokens`

**POST** endpoint for token counting.

**Request Body:**
- Anthropic format request

**Response:**
- Token count in Anthropic format

## 📝 Examples

See the `examples/` directory for sample usage patterns.

## 📝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🤝 Support

For issues or questions, please open an issue on GitHub.

## ❓ FAQ

### What is the Anthropic Adapter?
The Anthropic Adapter is an open-source API proxy that translates between Anthropic and OpenAI message formats, enabling seamless interoperability between different LLM providers.

### Why would I need this adapter?
You would need this adapter if you're using Anthropic SDKs but want to connect to OpenAI-compatible APIs, or if you want to switch between providers without changing your code.

### Can I use this with any OpenAI-compatible API?
Yes, the adapter is designed to work with any OpenAI-compatible API provider, not just the official OpenAI API.

### Does the adapter support streaming responses?
Yes, the adapter fully supports Server-Sent Events (SSE) streaming for real-time responses.

### How do I configure the adapter?
You can configure the adapter using environment variables in a `.env` file or by passing parameters directly to the CLI.

### What about v1/responses endpoint support?
Yes! The adapter now supports both `v1/chat/completions` and `v1/responses` endpoints. Simply set your `OPENAI_BASE_URL` to the desired endpoint and the adapter will automatically detect and use the appropriate transformation logic.


---

> **Note**: This project is constantly evolving. Check back for updates!
