"""
Tantra Web Integration Module
Web scraping, search, and data collection capabilities
"""

from .web_scraper import WebScraper, ScrapingConfig
from .web_search import WebSearch, SearchConfig
from .data_collector import WebDataCollector
from .content_processor import ContentProcessor

__all__ = [
    'WebScraper',
    'ScrapingConfig',
    'WebSearch', 
    'SearchConfig',
    'WebDataCollector',
    'ContentProcessor'
]