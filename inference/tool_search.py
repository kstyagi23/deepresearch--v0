import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import os


TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def tavily_search(self, query: str):
        """Perform a web search using Tavily API."""
        url = "https://api.tavily.com/search"
        
        headers = {
            'Authorization': f'Bearer {TAVILY_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "query": query,
            "max_results": 10,
            "search_depth": "basic",
            "topic": "general"
        }
        
        for i in range(5):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Tavily search timeout, return None. Please try again later."
                continue
        
        results = response.json()

        try:
            if "results" not in results or len(results["results"]) == 0:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            for page in results["results"]:
                idx += 1
                
                # Extract published date if available
                date_published = ""
                if "published_date" in page and page["published_date"]:
                    date_published = "\nDate published: " + page["published_date"]

                # Content from Tavily
                snippet = ""
                if "content" in page and page["content"]:
                    snippet = "\n" + page["content"]

                redacted_version = f"{idx}. [{page['title']}]({page['url']}){date_published}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

            content = f"A web search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # Single query
            response = self.tavily_search(query)
        else:
            # Multiple queries
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.tavily_search(q))
            response = "\n=======\n".join(responses)
            
        return response
