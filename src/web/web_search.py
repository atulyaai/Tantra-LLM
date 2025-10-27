"""
Web Search Module for Tantra
Real-time web search and information retrieval
"""

import requests
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import urllib.parse

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for web search"""
    
    # Search engines
    primary_engine: str = "google"  # google, bing, duckduckgo
    fallback_engines: List[str] = None
    
    # API settings
    google_api_key: str = ""
    google_search_engine_id: str = ""
    bing_api_key: str = ""
    
    # Search parameters
    max_results: int = 10
    language: str = "en"
    region: str = "us"
    safe_search: str = "moderate"  # off, moderate, strict
    
    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 5000
    content_types: List[str] = None  # text, pdf, doc, etc.
    
    # Rate limiting
    requests_per_minute: int = 60
    delay_between_requests: float = 1.0
    
    def __post_init__(self):
        """Initialize default values"""
        if self.fallback_engines is None:
            self.fallback_engines = ["bing", "duckduckgo"]
        if self.content_types is None:
            self.content_types = ["text"]


class WebSearch:
    """Real-time web search with multiple engines"""
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.session = requests.Session()
        self.request_times = []
        
        logger.info("WebSearch initialized")
    
    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search for a query using primary engine"""
        max_results = max_results or self.config.max_results
        
        try:
            # Rate limiting
            self._enforce_rate_limit()
            
            # Search with primary engine
            results = self._search_engine(query, self.config.primary_engine, max_results)
            
            if results:
                logger.info(f"Found {len(results)} results for '{query}'")
                return results
            else:
                # Try fallback engines
                for engine in self.config.fallback_engines:
                    logger.info(f"Trying fallback engine: {engine}")
                    results = self._search_engine(query, engine, max_results)
                    if results:
                        return results
                
                logger.warning(f"No results found for '{query}'")
                return []
                
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []
    
    def search_multiple_queries(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Search multiple queries and return combined results"""
        results = {}
        
        for query in queries:
            logger.info(f"Searching for: {query}")
            results[query] = self.search(query)
            time.sleep(self.config.delay_between_requests)
        
        return results
    
    def search_and_scrape(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search and scrape content from results"""
        from .web_scraper import WebScraper, ScrapingConfig
        
        # Get search results
        search_results = self.search(query, max_results)
        
        if not search_results:
            return []
        
        # Initialize scraper
        scraper_config = ScrapingConfig(
            extract_text=True,
            extract_metadata=True,
            min_content_length=self.config.min_content_length,
            max_content_length=self.config.max_content_length
        )
        scraper = WebScraper(scraper_config)
        
        # Scrape content from top results
        scraped_results = []
        for result in search_results[:5]:  # Limit to top 5 for scraping
            try:
                scraped = scraper.scrape_url(result['url'])
                if scraped['success']:
                    # Combine search and scraped data
                    combined = {
                        'search_rank': result.get('rank', 0),
                        'search_title': result.get('title', ''),
                        'search_snippet': result.get('snippet', ''),
                        'url': result['url'],
                        'content': scraped['content'],
                        'metadata': scraped['metadata'],
                        'content_length': len(scraped['content'])
                    }
                    scraped_results.append(combined)
                    
            except Exception as e:
                logger.error(f"Failed to scrape {result['url']}: {e}")
                continue
        
        return scraped_results
    
    def _search_engine(self, query: str, engine: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using specific engine"""
        if engine == "google":
            return self._search_google(query, max_results)
        elif engine == "bing":
            return self._search_bing(query, max_results)
        elif engine == "duckduckgo":
            return self._search_duckduckgo(query, max_results)
        else:
            logger.warning(f"Unknown search engine: {engine}")
            return []
    
    def _search_google(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        if not self.config.google_api_key or not self.config.google_search_engine_id:
            logger.warning("Google API credentials not configured")
            return self._search_google_web(query, max_results)
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.config.google_api_key,
                'cx': self.config.google_search_engine_id,
                'q': query,
                'num': min(max_results, 10),
                'lr': f"lang_{self.config.language}",
                'safe': self.config.safe_search
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for i, item in enumerate(data.get('items', [])):
                result = {
                    'rank': i + 1,
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'engine': 'google'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Google API search failed: {e}")
            return self._search_google_web(query, max_results)
    
    def _search_google_web(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback Google web search (simplified)"""
        # This is a simplified implementation
        # In practice, you'd use proper web scraping or APIs
        
        logger.info(f"Using Google web search for: {query}")
        
        # Mock results for demonstration
        return [
            {
                'rank': 1,
                'title': f"Search result for {query}",
                'url': f"https://example.com/search?q={urllib.parse.quote(query)}",
                'snippet': f"This is a search result for the query: {query}",
                'engine': 'google_web'
            }
        ]
    
    def _search_bing(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Bing Search API"""
        if not self.config.bing_api_key:
            logger.warning("Bing API key not configured")
            return []
        
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {
                'Ocp-Apim-Subscription-Key': self.config.bing_api_key
            }
            params = {
                'q': query,
                'count': min(max_results, 50),
                'mkt': f"{self.config.language}-{self.config.region}",
                'safeSearch': self.config.safe_search
            }
            
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for i, item in enumerate(data.get('webPages', {}).get('value', [])):
                result = {
                    'rank': i + 1,
                    'title': item.get('name', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('snippet', ''),
                    'engine': 'bing'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (simplified)"""
        # DuckDuckGo doesn't have a public API, so this is a simplified implementation
        
        logger.info(f"Using DuckDuckGo search for: {query}")
        
        # Mock results for demonstration
        return [
            {
                'rank': 1,
                'title': f"DuckDuckGo result for {query}",
                'url': f"https://duckduckgo.com/?q={urllib.parse.quote(query)}",
                'snippet': f"DuckDuckGo search result for: {query}",
                'engine': 'duckduckgo'
            }
        ]
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're over the limit
        if len(self.request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Add current request time
        self.request_times.append(current_time)
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions for a query"""
        try:
            # This would typically use a search suggestions API
            # For now, return a simple implementation
            
            suggestions = [
                f"{query} tutorial",
                f"{query} guide",
                f"{query} examples",
                f"{query} documentation",
                f"how to {query}"
            ]
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get suggestions for '{query}': {e}")
            return []
    
    def get_trending_topics(self, category: str = "general") -> List[str]:
        """Get trending topics for a category"""
        try:
            # This would typically use a trending topics API
            # For now, return mock data
            
            trending = {
                "general": [
                    "artificial intelligence",
                    "machine learning",
                    "python programming",
                    "web development",
                    "data science"
                ],
                "tech": [
                    "ChatGPT",
                    "GPT-4",
                    "transformer models",
                    "neural networks",
                    "deep learning"
                ],
                "news": [
                    "latest technology news",
                    "AI developments",
                    "programming updates",
                    "tech industry news"
                ]
            }
            
            return trending.get(category, trending["general"])
            
        except Exception as e:
            logger.error(f"Failed to get trending topics: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'total_requests': len(self.request_times),
            'requests_last_minute': len([t for t in self.request_times if time.time() - t < 60]),
            'rate_limit': self.config.requests_per_minute,
            'primary_engine': self.config.primary_engine,
            'fallback_engines': self.config.fallback_engines
        }