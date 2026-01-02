"""
Web search integration module for LLM-API-Key-Proxy.

This module provides intelligent web search capabilities with fallback chain:
Tavily → Brave Search → DuckDuckGo

The search is only activated when user prompts justify it (questions about recent info, etc.)
"""

import asyncio
import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
import httpx
import json
import time
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)


class WebSearchResult:
    """Represents a web search result."""
    
    def __init__(self, title: str, content: str, url:str = "", source: str = ""):
        self.title = title
        self.content = content
        self.url = url
        self.source = source
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source
        }


class WebSearchProvider(ABC):
    """Abstract base class for web search providers."""
    
    def __init__(self, api_key: Optional[str] = None, enabled: bool = True):
        self.api_key = api_key
        self.enabled = enabled
        self.timeout = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Perform a web search and return results."""
        pass
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def is_available(self) -> bool:
        """Check if this provider is available (enabled and configured)."""
        return self.enabled


class TavilyProvider(WebSearchProvider):
    """Tavily web search provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, enabled=bool(api_key))
        self.base_url = "https://api.tavily.com"
    
    def is_available(self) -> bool:
        return super().is_available() and self.api_key is not None
    
    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Search using Tavily API."""
        if not self.is_available():
            logger.info("Tavily: Not available (no API key)")
            return []
        
        try:
            logger.info(f"Tavily: Searching for '{query}'")
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False
            }
            
            response = await self.client.post(
                f"{self.base_url}/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for result in data.get("results", []):
                results.append(WebSearchResult(
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    url=result.get("url", ""),
                    source="tavily"
                ))
            
            logger.info(f"Tavily: Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Tavily: Search failed - {e}")
            return []


class BraveProvider(WebSearchProvider):
    """Brave Search API provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, enabled=bool(api_key))
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
    
    def is_available(self) -> bool:
        return super().is_available() and self.api_key is not None
    
    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Search using Brave Search API."""
        if not self.is_available():
            logger.info("Brave: Not available (no API key)")
            return []
        
        try:
            logger.info(f"Brave: Searching for '{query}'")
            
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "count": max_results,
                "safesearch": "moderate"
            }
            
            response = await self.client.get(
                self.base_url,
                params=params,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for result in data.get("web", {}).get("results", []):
                results.append(WebSearchResult(
                    title=result.get("title", ""),
                    content=result.get("description", ""),
                    url=result.get("url", ""),
                    source="brave"
                ))
            
            logger.info(f"Brave: Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Brave: Search failed - {e}")
            return []


class DuckDuckGoProvider(WebSearchProvider):
    """DuckDuckGo web search provider (no API key required)."""
    
    def __init__(self):
        enabled = os.getenv("DUCKDUCKGO_ENABLED", "true").lower() == "true"
        super().__init__(None, enabled=enabled)
    
    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Search using DuckDuckGo Instant Answer API."""
        if not self.is_available():
            logger.info("DuckDuckGo: Not enabled")
            return []
        
        try:
            logger.info(f"DuckDuckGo: Searching for '{query}'")
            
            # Use DuckDuckGo's HTML endpoint since they don't have a proper search API
            search_url = f"https://duckduckgo.com/html/?q={quote(query)}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = await self.client.get(search_url, headers=headers)
            response.raise_for_status()
            
            html_content = response.text
            results = []
            
            # Simple regex parsing of DuckDuckGo HTML results
            # This is fragile but works for basic needs
            result_blocks = re.findall(
                r'<a class="result__a"[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>',
                html_content,
                re.DOTALL
            )
            
            for i, (title, snippet) in enumerate(result_blocks[:max_results]):
                # Clean HTML tags
                clean_title = re.sub(r'<.*?>', '', title)
                clean_snippet = re.sub(r'<.*?>', '', snippet)
                
                results.append(WebSearchResult(
                    title=clean_title.strip(),
                    content=clean_snippet.strip(),
                    url="",  # DuckDuckGo HTML doesn't easily give us URLs
                    source="duckduckgo"
                ))
            
            logger.info(f"DuckDuckGo: Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo: Search failed - {e}")
            return []


class WebSearchDetector:
    """Detects when web search is needed based on user prompts."""
    
    def __init__(self):
        self.web_search_keywords = {
            # Question words
            "what", "when", "where", "who", "why", "how", "which", "whom", "whose",
            # Time indicators
            "current", "latest", "recent", "today", "yesterday", "this week", "this month",
            "now", "update", "news", "breaking", "trending",
            # Information seeking
            "happened", "update on", "information about", "details about", "status of",
            "price of", "value of", "rate of", "cost of",
            # Current events
            "weather", "stock", "bitcoin", "crypto", "price", "market", "election",
            "score", "game", "match", "result", "breaking news"
        }
        
        # Negative patterns - don't search for these
        self.no_search_patterns = [
            r"write.*code", r"python\s+(function|class|script)", r"javascript\s+code",
            r"debug.*code", r"fix.*error", r"explain.*code",
            r"quantum\s+computing", r"machine\s+learning", r"algorithm",
            r"my\s+(preference|opinion|thought)", r"i think", r"personal",
            r"hypothetical", r"theoretical", r"philosophical",
            r"write.*essay", r"compose.*poem", r"generate.*text"
        ]
    
    def should_search(self, prompt: str) -> bool:
        """
        Determine if web search is needed based on the prompt.
        
        Args:
            prompt: The user's prompt
            
        Returns:
            True if web search should be performed
        """
        if not prompt:
            return False
        
        prompt_lower = prompt.lower().strip()
        
        # Check negative patterns first
        for pattern in self.no_search_patterns:
            if re.search(pattern, prompt_lower):
                logger.debug(f"Web search skipped due to negative pattern: {pattern}")
                return False
        
        # Check for web search keywords
        words = prompt_lower.split()
        for i in range(len(words)):
            # Check single words
            if words[i] in self.web_search_keywords:
                logger.debug(f"Web search triggered by keyword: '{words[i]}'")
                return True
            
            # Check phrases (bigrams)
            if i < len(words) - 1:
                phrase = f"{words[i]} {words[i+1]}"
                if phrase in self.web_search_keywords:
                    logger.debug(f"Web search triggered by phrase: '{phrase}'")
                    return True
        
        # Special cases
        if prompt_lower.startswith(("what's ", "whats ", "what is ", "what are ", 
                                   "who is ", "who was ", "where is ", "when was ")):
            logger.debug("Web search triggered by question format")
            return True
        
        return False


class WebSearchManager:
    """Manages web search with provider fallback chain."""
    
    def __init__(self):
        self.detector = WebSearchDetector()
        
        # Initialize providers in order of preference
        tavily_key = os.getenv("TAVILY_API_KEY")
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
        
        self.providers = []
        
        # Tavily (primary)
        if tavily_key:
            self.providers.append(TavilyProvider(tavily_key))
        
        # Brave (secondary)
        if brave_key:
            self.providers.append(BraveProvider(brave_key))
        
        # DuckDuckGo (fallback)
        self.providers.append(DuckDuckGoProvider())
        
        logger.info(f"Web search initialized with {len(self.providers)} providers")
    
    async def search_if_needed(
        self, 
        prompt: str, 
        use_web_search: Optional[bool] = None,
        max_results: int = 5
    ) -> Tuple[bool, List[WebSearchResult], str]:
        """
        Perform web search if needed and allowed.
        
        Args:
            prompt: User's prompt
            use_web_search: Override for auto-detection (True=force, False=disable, None=auto)
            max_results: Maximum results to return
            
        Returns:
            Tuple of (was_search_used, results, provider_name)
        """
        # Determine if search should be used
        should_search = False
        if use_web_search is True:
            should_search = True
            logger.info("Web search: Forced by user request")
        elif use_web_search is False:
            should_search = False
            logger.info("Web search: Disabled by user request")
        else:
            # Auto-detect
            auto_detect = os.getenv("WEB_SEARCH_AUTO_DETECT", "true").lower() == "true"
            if auto_detect:
                should_search = self.detector.should_search(prompt)
                logger.info(f"Web search: Auto-detection {'triggered' if should_search else 'not triggered'}")
            else:
                should_search = False
                logger.info("Web search: Auto-detection disabled")
        
        if not should_search:
            return False, [], ""
        
        # Try providers in order
        for provider in self.providers:
            if not provider.is_available():
                continue
            
            try:
                results = await provider.search(prompt, max_results)
                if results:
                    logger.info(f"Web search successful via {provider.__class__.__name__}")
                    return True, results, provider.__class__.__name__.lower().replace("provider", "")
            except Exception as e:
                logger.error(f"Web search failed with {provider.__class__.__name__}: {e}")
                continue
        
        logger.warning("Web search: All providers failed")
        return False, [], ""
    
    async def format_results_for_prompt(self, results: List[WebSearchResult]) -> str:
        """
        Format search results for inclusion in LLM prompt.
        
        Returns a concise string that can be added to the system prompt.
        """
        if not results:
            return ""
        
        formatted_results = []
        formatted_results.append("Based on web search results:")
        formatted_results.append("")
        
        for i, result in enumerate(results[:3], 1):  # Limit to top 3 results
            result_text = f"{i}. {result.title}"
            if result.content:
                # Truncate content to avoid huge prompts
                content_preview = result.content[:200]
                if len(result.content) > 200:
                    content_preview += "..."
                result_text += f"\n   {content_preview}"
            if result.url:
                result_text += f"\n   Source: {result.url}"
            formatted_results.append(result_text)
        
        return "\n".join(formatted_results)
    
    async def close(self):
        """Close all provider clients."""
        await asyncio.gather(*[provider.close() for provider in self.providers])


# Global singleton instance
_web_search_manager: Optional[WebSearchManager] = None


def get_web_search_manager() -> WebSearchManager:
    """Get the global web search manager instance."""
    global _web_search_manager
    if _web_search_manager is None:
        _web_search_manager = WebSearchManager()
    return _web_search_manager
