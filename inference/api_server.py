"""
DeepResearch API Server
FastAPI server with streaming research endpoint.
"""

import os
import json
import json5
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from research_agent import ResearchAgent, ResearchConfig, ResearchEvent


# === Request/Response Models ===

class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ResearchRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages, last user message is the research question")
    temperature: Optional[float] = Field(0.6, ge=0, le=2, description="Sampling temperature")
    max_llm_calls: Optional[int] = Field(100, ge=1, le=500, description="Maximum LLM calls")


class ResearchResponse(BaseModel):
    question: str
    answer: Optional[str]
    termination: str
    rounds: Optional[int] = None
    error: Optional[str] = None


# === Application Setup ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 DeepResearch API Server starting...")
    print(f"   Model: {os.getenv('MAIN_MODEL_NAME', 'not set')}")
    print(f"   Base URL: {os.getenv('MAIN_MODEL_BASE_URL', 'not set')}")
    yield
    print("👋 DeepResearch API Server shutting down...")


app = FastAPI(
    title="DeepResearch API",
    description="Deep information-seeking research agent API with streaming support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_stream_event(event: ResearchEvent) -> dict:
    """
    Format event into the standardized streaming format:
    {
        "type": "tool" | "text",
        "text": "..." (if text),
        "tool": "..." (if tool),
        "link": "..." (optional, if tool has URL),
        "args": {...} (optional, tool arguments)
    }
    """
    # Tool-related events
    if event.type == "tool_call":
        result = {"type": "tool"}
        
        # Parse the tool call to extract name and args
        try:
            tool_data = json5.loads(event.content)
            result["tool"] = tool_data.get("name", "unknown")
            
            args = tool_data.get("arguments", {})
            if args:
                result["args"] = args
                
                # Extract link if present (from visit or search tools)
                if "url" in args:
                    url = args["url"]
                    if isinstance(url, list) and url:
                        result["link"] = url[0]
                    elif isinstance(url, str):
                        result["link"] = url
                        
        except:
            # Python interpreter or unparseable
            if "python" in event.content.lower():
                result["tool"] = "PythonInterpreter"
            else:
                result["tool"] = "unknown"
                result["args"] = event.content
        
        return result
    
    elif event.type == "tool_response":
        # Tool results as text
        return {
            "type": "text",
            "text": f"[Tool Result] {event.content}"
        }
    
    # Text-based events
    elif event.type == "thinking":
        return {
            "type": "text",
            "text": event.content
        }
    
    elif event.type == "answer":
        return {
            "type": "text",
            "text": event.content,
            "is_answer": True
        }
    
    elif event.type == "status":
        return {
            "type": "text",
            "text": f"[Status] {event.content}"
        }
    
    elif event.type == "error":
        return {
            "type": "text",
            "text": f"[Error] {event.content}",
            "is_error": True
        }
    
    else:
        return {
            "type": "text",
            "text": event.content
        }


# === Endpoints ===

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "deepresearch"}


@app.post("/api/v1/research")
async def research(request: ResearchRequest):
    """
    Conduct deep research on a question.
    
    Returns streaming JSON responses in format:
    ```json
    {"type": "tool", "tool": "search", "args": {"query": ["..."]}}
    {"type": "tool", "tool": "visit", "link": "https://...", "args": {...}}
    {"type": "text", "text": "Thinking about the problem..."}
    {"type": "text", "text": "Final answer here", "is_answer": true}
    ```
    """
    # Extract the question from the last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in messages")
    
    question = user_messages[-1].content
    
    # Create config
    config = ResearchConfig(
        temperature=request.temperature or 0.6,
        max_llm_calls=request.max_llm_calls or 100,
    )
    
    # Create agent
    agent = ResearchAgent(config)
    
    # Streaming response
    async def generate():
        try:
            gen = agent.research_stream(question)
            try:
                while True:
                    event = next(gen)
                    event_data = format_stream_event(event)
                    yield json.dumps(event_data) + "\n"
            except StopIteration as e:
                # Send final result marker
                final_result = e.value
                if final_result and final_result.get("answer"):
                    yield json.dumps({
                        "type": "text",
                        "text": final_result["answer"],
                        "is_answer": True,
                        "is_final": True
                    }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "text",
                "text": f"Error: {str(e)}",
                "is_error": True
            }) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/v1/research/sync")
async def research_sync(request: ResearchRequest):
    """
    Non-streaming research endpoint.
    Returns complete result after research is done.
    """
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in messages")
    
    question = user_messages[-1].content
    
    config = ResearchConfig(
        temperature=request.temperature or 0.6,
        max_llm_calls=request.max_llm_calls or 100,
    )
    
    agent = ResearchAgent(config)
    
    try:
        result = await agent.research_async(question)
        return ResearchResponse(
            question=result.get("question", question),
            answer=result.get("answer"),
            termination=result.get("termination", "unknown"),
            rounds=result.get("rounds"),
            error=result.get("error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/research/fake-stream")
async def fake_stream():
    """
    Fake streaming endpoint for testing.
    Returns random research-like data as a stream.
    """
    import random
    import string
    
    fake_tools = ["search", "visit", "google_scholar", "parse_file", "PythonInterpreter"]
    fake_urls = [
        "https://arxiv.org/abs/2401.12345",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://nature.com/articles/s41586-024-07890",
        "https://github.com/example/repo",
        "https://docs.python.org/3/library/asyncio.html",
    ]
    
    thinking_phrases = [
        "Let me analyze this problem step by step...",
        "I'm searching for more information on this topic...",
        "Based on the search results, I found several relevant sources...",
        "Processing the retrieved data...",
        "Synthesizing information from multiple sources...",
        "Cross-referencing the findings with academic literature...",
        "Evaluating the credibility of different sources...",
        "Formulating a comprehensive response...",
    ]
    
    async def generate():
        # Initial status
        yield json.dumps({"type": "text", "text": "[Status] Starting fake research..."}) + "\n"
        await asyncio.sleep(0.5)
        
        # Random number of rounds
        num_rounds = random.randint(3, 6)
        
        for round_num in range(num_rounds):
            # Thinking text
            yield json.dumps({"type": "text", "text": random.choice(thinking_phrases)}) + "\n"
            await asyncio.sleep(random.uniform(0.3, 0.8))
            
            # Tool call
            tool = random.choice(fake_tools)
            tool_event = {"type": "tool", "tool": tool}
            
            if tool == "search":
                queries = [f"query_{random.randint(1, 100)}" for _ in range(random.randint(1, 3))]
                tool_event["args"] = {"query": queries}
            elif tool == "visit":
                url = random.choice(fake_urls)
                tool_event["link"] = url
                tool_event["args"] = {"url": url, "goal": "Extract relevant information"}
            elif tool == "google_scholar":
                tool_event["args"] = {"query": [f"academic research {random.randint(1, 50)}"]}
            
            yield json.dumps(tool_event) + "\n"
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Tool result
            random_content = ''.join(random.choices(string.ascii_letters + ' ', k=random.randint(100, 300)))
            yield json.dumps({"type": "text", "text": f"[Tool Result] {random_content}"}) + "\n"
            await asyncio.sleep(random.uniform(0.2, 0.5))
        
        # Final answer
        final_answer = (
            "Based on my comprehensive research, here are the key findings:\n\n"
            "1. This is a randomly generated fake response for testing streaming.\n"
            "2. The actual research would provide real, synthesized information.\n"
            "3. Use the /api/v1/research endpoint for actual research queries.\n\n"
            f"Research completed in {num_rounds} rounds."
        )
        yield json.dumps({"type": "text", "text": final_answer, "is_answer": True, "is_final": True}) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 12123))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"\n🔬 Starting DeepResearch API Server on {host}:{port}\n")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        log_level="info",
    )
