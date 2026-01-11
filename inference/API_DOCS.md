# DeepResearch API Documentation

A FastAPI-based REST API for conducting deep information-seeking research using AI agents.

---

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
# - MAIN_MODEL_API_KEY
# - MAIN_MODEL_BASE_URL
# - MAIN_MODEL_NAME
# - TOKENIZER
```

### 2. Start the Server

```powershell
cd inference
python api_server.py
```

Server runs at `http://localhost:8000`

---

## Endpoints

### Health Check

```
GET /health
```

**Response:**

```json
{ "status": "ok", "service": "deepresearch" }
```

---

### Research (Streaming)

```
POST /api/v1/research
```

Conducts deep research with streaming responses.

**Request Body:**

```json
{
  "messages": [{ "role": "user", "content": "Your research question here" }],
  "temperature": 0.6,
  "max_llm_calls": 100
}
```

| Field           | Type  | Default  | Description                                      |
| --------------- | ----- | -------- | ------------------------------------------------ |
| `messages`      | array | required | Chat messages, last user message is the question |
| `temperature`   | float | 0.6      | Sampling temperature (0-2)                       |
| `max_llm_calls` | int   | 100      | Maximum LLM calls before stopping                |

**Response:** Streaming newline-delimited JSON (`application/x-ndjson`)

Each line is a JSON object:

```json
{"type": "text", "text": "[Status] Starting research..."}
{"type": "text", "text": "Let me search for information..."}
{"type": "tool", "tool": "search", "args": {"query": ["quantum computing breakthroughs 2024"]}}
{"type": "text", "text": "[Tool Result] Found 10 results..."}
{"type": "tool", "tool": "visit", "link": "https://example.com/article", "args": {"url": "...", "goal": "..."}}
{"type": "text", "text": "Based on my research...", "is_answer": true}
```

---

### Research (Synchronous)

```
POST /api/v1/research/sync
```

Same request format, but waits for complete result.

**Response:**

```json
{
  "question": "Your question",
  "answer": "The complete answer",
  "termination": "completed",
  "rounds": 5,
  "error": null
}
```

---

## Streaming Response Format

### Text Events

```json
{
  "type": "text",
  "text": "Content being streamed",
  "is_answer": true, // Optional: marks final answer
  "is_error": true // Optional: marks error
}
```

### Tool Events

```json
{
  "type": "tool",
  "tool": "tool_name",
  "link": "https://...", // Optional: URL being accessed
  "args": {} // Optional: tool arguments
}
```

**Available Tools:**

| Tool                | Description                   |
| ------------------- | ----------------------------- |
| `search`            | Web search via Google         |
| `visit`             | Visit and summarize web pages |
| `google_scholar`    | Academic paper search         |
| `parse_file`        | Parse PDF, DOCX, etc.         |
| `PythonInterpreter` | Execute Python code           |

---

## Example Usage

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is CRISPR?"}]}'
```

### Python

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/api/v1/research",
    json={"messages": [{"role": "user", "content": "Explain quantum computing"}]},
    stream=True
)

for line in response.iter_lines():
    if line:
        event = json.loads(line)
        if event["type"] == "text":
            print(event["text"])
        elif event["type"] == "tool":
            print(f"[Using {event['tool']}]")
```

### JavaScript

```javascript
const response = await fetch("http://localhost:8000/api/v1/research", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "What is machine learning?" }],
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { value, done } = await reader.read();
  if (done) break;

  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (line.trim()) {
      const event = JSON.parse(line);
      console.log(event);
    }
  }
}
```

---

## Python SDK

Use the `ResearchAgent` class directly:

```python
from research_agent import ResearchAgent

agent = ResearchAgent()

# Simple usage
result = agent.research("What are black holes?")
print(result["answer"])

# Streaming
for event in agent.research_stream("Explain DNA"):
    print(f"[{event.type}] {event.content[:100]}")
```

---

## Configuration

Environment variables in `.env`:

| Variable               | Description                    |
| ---------------------- | ------------------------------ |
| `MAIN_MODEL_API_KEY`   | API key for the LLM            |
| `MAIN_MODEL_BASE_URL`  | API endpoint URL               |
| `MAIN_MODEL_NAME`      | Model name/ID                  |
| `TOKENIZER`            | HuggingFace tokenizer ID       |
| `MAX_LLM_CALL_PER_RUN` | Max LLM calls (default: 100)   |
| `PORT`                 | Server port (default: 8000)    |
| `HOST`                 | Server host (default: 0.0.0.0) |
