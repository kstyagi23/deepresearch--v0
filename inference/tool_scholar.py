import os
import json
import requests
from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor


TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')


@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Leverage web search to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries for academic/scholarly content."
                },
            },
        "required": ["query"],
    }

    def tavily_scholar_search(self, query: str):
        """Perform a scholar-focused search using Tavily API with academic keywords."""
        url = "https://api.tavily.com/search"
        
        headers = {
            'Authorization': f'Bearer {TAVILY_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Enhance query with academic focus
        academic_query = f"{query} academic paper research study"
        
        payload = {
            "query": academic_query,
            "max_results": 10,
            "search_depth": "advanced",  # Use advanced for better academic results
            "topic": "general",
            "include_domains": [
                "scholar.google.com",
                "arxiv.org",
                "pubmed.ncbi.nlm.nih.gov",
                "researchgate.net",
                "academia.edu",
                "sciencedirect.com",
                "springer.com",
                "ieee.org",
                "nature.com",
                "science.org",
                "jstor.org",
                "semanticscholar.org"
            ]
        }
        
        for i in range(5):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Scholar search timeout, return None. Please try again later."
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
                    date_published = "\nDate published: " + str(page["published_date"])

                # Source info from URL domain
                source_info = ""
                if "url" in page:
                    from urllib.parse import urlparse
                    domain = urlparse(page["url"]).netloc
                    source_info = f"\nSource: {domain}"

                # Content snippet
                snippet = ""
                if "content" in page and page["content"]:
                    snippet = "\n" + page["content"]

                redacted_version = f"{idx}. [{page['title']}]({page['url']}){source_info}{date_published}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "") 
                web_snippets.append(redacted_version)

            content = f"A scholar search for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."


    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            response = self.tavily_scholar_search(query)
        else:
            assert isinstance(query, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(self.tavily_scholar_search, query))
            response = "\n=======\n".join(response)
        return response
