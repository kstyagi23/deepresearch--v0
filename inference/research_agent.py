"""
DeepResearch - Callable Research Agent
A simple, callable interface for the DeepResearch agent.
"""

import os
import json
import json5
import time
import random
import asyncio
import datetime
from typing import Optional, List, Dict, Any, Generator, AsyncGenerator
from dataclasses import dataclass, field
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import tools
from tool_file import FileParser
from tool_scholar import Scholar
from tool_python import PythonInterpreter
from tool_search import Search
from tool_visit import Visit
from prompt import SYSTEM_PROMPT

# Configuration from environment
CLOUD_API_KEY = os.getenv("MAIN_MODEL_API_KEY", "")
CLOUD_BASE_URL = os.getenv("MAIN_MODEL_BASE_URL", "")
CLOUD_MODEL_NAME = os.getenv("MAIN_MODEL_NAME", "")
TOKENIZER_MODEL_ID = os.getenv("TOKENIZER", "")
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALL_PER_RUN", 100))
MAX_TIME_SECONDS = int(os.getenv("MAX_TIME_SECONDS", 9000))  # 150 minutes default


@dataclass
class ResearchConfig:
    """Configuration for the research agent."""
    temperature: float = 0.6
    top_p: float = 0.95
    presence_penalty: float = 1.1
    max_tokens: int = 10000
    max_context_tokens: int = 110 * 1024
    max_llm_calls: int = MAX_LLM_CALLS
    max_time_seconds: int = MAX_TIME_SECONDS
    timeout: float = 600.0


@dataclass
class ResearchEvent:
    """Event emitted during research."""
    type: str  # "thinking", "tool_call", "tool_response", "answer", "error", "status"
    content: str
    round: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchAgent:
    """
    Callable DeepResearch agent for conducting deep information-seeking research.
    
    Usage:
        agent = ResearchAgent()
        
        # Simple usage - get final result
        result = agent.research("What are the latest breakthroughs in quantum computing?")
        print(result["answer"])
        
        # Streaming usage - get events as they happen
        for event in agent.research_stream("Explain CRISPR gene editing"):
            print(f"[{event.type}] {event.content[:100]}...")
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None):
        """Initialize the research agent."""
        self.config = config or ResearchConfig()
        self._tokenizer = None
        
        # Initialize tools
        self.tools = {
            "parse_file": FileParser(),
            "google_scholar": Scholar(),
            "visit": Visit(),
            "search": Search(),
            "PythonInterpreter": PythonInterpreter(),
        }
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=CLOUD_API_KEY,
            base_url=CLOUD_BASE_URL,
            timeout=self.config.timeout,
        )
    
    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)
        return self._tokenizer
    
    def _today_date(self) -> str:
        return datetime.date.today().strftime("%Y-%m-%d")
    
    def _count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in messages."""
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer(full_prompt, return_tensors="pt")
        return len(tokens["input_ids"][0])
    
    def _call_llm(self, messages: List[Dict], max_tries: int = 10) -> str:
        """Call the LLM with retry logic."""
        base_sleep_time = 1
        
        for attempt in range(max_tries):
            try:
                response = self.client.chat.completions.create(
                    model=CLOUD_MODEL_NAME,
                    messages=messages,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    presence_penalty=self.config.presence_penalty,
                )
                content = response.choices[0].message.content
                
                if content and content.strip():
                    return content.strip()
                    
            except (APIError, APIConnectionError, APITimeoutError) as e:
                if attempt == max_tries - 1:
                    raise
                    
            except Exception as e:
                if attempt == max_tries - 1:
                    raise
            
            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                time.sleep(sleep_time)
        
        raise Exception("LLM call failed after all retries")
    
    def _execute_tool(self, tool_call_str: str, full_content: str) -> str:
        """Execute a tool call and return the result."""
        try:
            if "python" in tool_call_str.lower():
                # Python interpreter has special format
                code_raw = full_content.split('<tool_call>')[1].split('</tool_call>')[0]
                code_raw = code_raw.split('<code>')[1].split('</code>')[0].strip()
                return self.tools['PythonInterpreter'].call(code_raw)
            else:
                tool_call = json5.loads(tool_call_str)
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('arguments', {})
                
                if tool_name not in self.tools:
                    return f"Error: Tool '{tool_name}' not found"
                
                tool_args["params"] = tool_args
                
                if tool_name == "parse_file":
                    params = {"files": tool_args.get("files", [])}
                    result = asyncio.run(self.tools[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                    return str(result) if not isinstance(result, str) else result
                else:
                    return self.tools[tool_name].call(tool_args)
                    
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def research_stream(self, question: str) -> Generator[ResearchEvent, None, Dict[str, Any]]:
        """
        Conduct research with streaming events.
        
        Args:
            question: The research question
            
        Yields:
            ResearchEvent objects for each step
            
        Returns:
            Final result dictionary
        """
        start_time = time.time()
        system_prompt = SYSTEM_PROMPT + self._today_date()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        yield ResearchEvent(type="status", content="Starting research...", round=0)
        
        round_num = 0
        calls_remaining = self.config.max_llm_calls
        
        while calls_remaining > 0:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.config.max_time_seconds:
                yield ResearchEvent(type="error", content="Research timeout reached", round=round_num)
                return {
                    "question": question,
                    "answer": None,
                    "messages": messages,
                    "termination": "timeout",
                    "error": "Research timeout reached"
                }
            
            round_num += 1
            calls_remaining -= 1
            
            yield ResearchEvent(
                type="status", 
                content=f"Round {round_num} - Generating response...",
                round=round_num
            )
            
            try:
                content = self._call_llm(messages)
            except Exception as e:
                yield ResearchEvent(type="error", content=str(e), round=round_num)
                return {
                    "question": question,
                    "answer": None,
                    "messages": messages,
                    "termination": "error",
                    "error": str(e)
                }
            
            # Clean content
            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]
            
            messages.append({"role": "assistant", "content": content.strip()})
            
            # Emit thinking if present
            if '<think>' in content and '</think>' in content:
                thinking = content.split('<think>')[1].split('</think>')[0]
                yield ResearchEvent(type="thinking", content=thinking.strip(), round=round_num)
            
            # Check for tool calls
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                
                yield ResearchEvent(
                    type="tool_call", 
                    content=tool_call_str.strip(),
                    round=round_num
                )
                
                result = self._execute_tool(tool_call_str, content)
                
                yield ResearchEvent(
                    type="tool_response",
                    content=result[:500] + "..." if len(result) > 500 else result,
                    round=round_num
                )
                
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})
            
            # Check for final answer
            if '<answer>' in content and '</answer>' in content:
                answer = content.split('<answer>')[1].split('</answer>')[0]
                yield ResearchEvent(type="answer", content=answer.strip(), round=round_num)
                
                return {
                    "question": question,
                    "answer": answer.strip(),
                    "messages": messages,
                    "termination": "completed",
                    "rounds": round_num
                }
            
            # Check token limit
            token_count = self._count_tokens(messages)
            if token_count > self.config.max_context_tokens:
                yield ResearchEvent(
                    type="status",
                    content="Context limit reached, forcing final answer...",
                    round=round_num
                )
                
                messages[-1]['content'] = (
                    "You have now reached the maximum context length. "
                    "Stop making tool calls and provide your best answer: "
                    "<think>your final thinking</think>\n<answer>your answer</answer>"
                )
                
                try:
                    content = self._call_llm(messages)
                    messages.append({"role": "assistant", "content": content.strip()})
                    
                    if '<answer>' in content and '</answer>' in content:
                        answer = content.split('<answer>')[1].split('</answer>')[0]
                        yield ResearchEvent(type="answer", content=answer.strip(), round=round_num)
                        
                        return {
                            "question": question,
                            "answer": answer.strip(),
                            "messages": messages,
                            "termination": "context_limit",
                            "rounds": round_num
                        }
                except Exception as e:
                    pass
                
                return {
                    "question": question,
                    "answer": None,
                    "messages": messages,
                    "termination": "context_limit_no_answer",
                    "rounds": round_num
                }
        
        # Exhausted LLM calls
        yield ResearchEvent(type="error", content="Max LLM calls reached", round=round_num)
        return {
            "question": question,
            "answer": None,
            "messages": messages,
            "termination": "max_calls_reached",
            "rounds": round_num
        }
    
    def research(self, question: str) -> Dict[str, Any]:
        """
        Conduct research and return final result.
        
        Args:
            question: The research question
            
        Returns:
            Dictionary with question, answer, messages, and metadata
        """
        result = None
        for event in self.research_stream(question):
            # The generator returns the final result
            pass
        
        # Get the return value from the generator
        gen = self.research_stream(question)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value
        
        return result
    
    async def research_async(self, question: str) -> Dict[str, Any]:
        """Async version of research."""
        return await asyncio.to_thread(self.research, question)
    
    async def research_stream_async(self, question: str) -> AsyncGenerator[ResearchEvent, None]:
        """Async streaming version of research."""
        def sync_generator():
            for event in self.research_stream(question):
                yield event
        
        for event in sync_generator():
            yield event
            await asyncio.sleep(0)  # Yield control


# Convenience function
def research(question: str, config: Optional[ResearchConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to conduct research.
    
    Args:
        question: The research question
        config: Optional configuration
        
    Returns:
        Research result dictionary
    """
    agent = ResearchAgent(config)
    return agent.research(question)


if __name__ == "__main__":
    # Example usage
    agent = ResearchAgent()
    
    print("=== DeepResearch Agent ===\n")
    question = input("Enter your research question: ").strip()
    
    if not question:
        question = "What is the capital of France?"
    
    print(f"\nResearching: {question}\n")
    print("-" * 50)
    
    for event in agent.research_stream(question):
        if event.type == "status":
            print(f"[STATUS] {event.content}")
        elif event.type == "thinking":
            print(f"[THINKING] {event.content[:200]}...")
        elif event.type == "tool_call":
            print(f"[TOOL] {event.content[:100]}...")
        elif event.type == "tool_response":
            print(f"[RESULT] {event.content[:200]}...")
        elif event.type == "answer":
            print(f"\n{'='*50}")
            print(f"ANSWER:\n{event.content}")
            print(f"{'='*50}")
        elif event.type == "error":
            print(f"[ERROR] {event.content}")
