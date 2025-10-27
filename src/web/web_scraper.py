"""
Web Scraping Module for Tantra
Advanced web scraping with content extraction and processing
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    """Configuration for web scraping"""
    
    # Request settings
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    timeout: int = 30
    max_retries: int = 3
    delay_between_requests: float = 1.0
    
    # Content extraction
    extract_text: bool = True
    extract_links: bool = True
    extract_images: bool = False
    extract_metadata: bool = True
    
    # Filtering
    min_content_length: int = 100
    max_content_length: int = 100000
    allowed_domains: List[str] = None
    blocked_domains: List[str] = None
    content_filters: List[str] = None
    
    # Storage
    save_raw_html: bool = False
    save_processed_content: bool = True
    output_format: str = "json"  # json, txt, markdown
    
    def __post_init__(self):
        """Initialize default values"""
        if self.allowed_domains is None:
            self.allowed_domains = []
        if self.blocked_domains is None:
            self.blocked_domains = []
        if self.content_filters is None:
            self.content_filters = []


class WebScraper:
    """Advanced web scraper with content extraction"""
    
    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.config.user_agent})
        
        # Content cache
        self.content_cache = {}
        
        logger.info("WebScraper initialized")
    
    def scrape_url(self, url: str, cache: bool = True) -> Dict[str, Any]:
        """Scrape a single URL and extract content"""
        
        # Check cache
        if cache and url in self.content_cache:
            logger.debug(f"Using cached content for {url}")
            return self.content_cache[url]
        
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            # Check domain restrictions
            if not self._is_allowed_domain(url):
                raise ValueError(f"Domain not allowed: {url}")
            
            # Make request
            response = self._make_request(url)
            
            # Parse content
            content = self._extract_content(response, url)
            
            # Cache result
            if cache:
                self.content_cache[url] = content
            
            # Add delay between requests
            time.sleep(self.config.delay_between_requests)
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                'url': url,
                'success': False,
                'error': str(e),
                'content': '',
                'metadata': {}
            }
    
    def scrape_urls(self, urls: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """Scrape multiple URLs"""
        results = []
        
        for i, url in enumerate(urls):
            logger.info(f"Scraping {i+1}/{len(urls)}: {url}")
            
            try:
                result = self.scrape_url(url)
                results.append(result)
                
                # Save intermediate results
                if self.config.save_processed_content:
                    self._save_content(result, f"scraped_{i+1}")
                    
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                results.append({
                    'url': url,
                    'success': False,
                    'error': str(e),
                    'content': '',
                    'metadata': {}
                })
        
        return results
    
    def scrape_sitemap(self, sitemap_url: str) -> List[Dict[str, Any]]:
        """Scrape all URLs from a sitemap"""
        try:
            # Get sitemap content
            response = self._make_request(sitemap_url)
            soup = BeautifulSoup(response.content, 'xml')
            
            # Extract URLs
            urls = []
            for loc in soup.find_all('loc'):
                urls.append(loc.text.strip())
            
            logger.info(f"Found {len(urls)} URLs in sitemap")
            
            # Scrape all URLs
            return self.scrape_urls(urls)
            
        except Exception as e:
            logger.error(f"Failed to scrape sitemap {sitemap_url}: {e}")
            return []
    
    def search_and_scrape(self, query: str, search_engine: str = "google",
                         max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for a query and scrape the results"""
        try:
            # Get search results
            search_results = self._get_search_results(query, search_engine, max_results)
            
            # Extract URLs
            urls = [result['url'] for result in search_results]
            
            # Scrape URLs
            scraped_content = self.scrape_urls(urls)
            
            # Combine search results with scraped content
            combined_results = []
            for i, (search_result, scraped) in enumerate(zip(search_results, scraped_content)):
                combined_result = {
                    'search_rank': i + 1,
                    'search_title': search_result.get('title', ''),
                    'search_snippet': search_result.get('snippet', ''),
                    **scraped
                }
                combined_results.append(combined_result)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to search and scrape for '{query}': {e}")
            return []
    
    def _make_request(self, url: str) -> requests.Response:
        """Make HTTP request with retries"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, timeout=self.config.timeout)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _extract_content(self, response: requests.Response, url: str) -> Dict[str, Any]:
        """Extract content from HTTP response"""
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        content = ""
        if self.config.extract_text:
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            content = soup.get_text()
            
            # Clean up text
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Apply content filters
            content = self._apply_content_filters(content)
        
        # Extract metadata
        metadata = {}
        if self.config.extract_metadata:
            metadata = self._extract_metadata(soup, response, url)
        
        # Extract links
        links = []
        if self.config.extract_links:
            links = self._extract_links(soup, url)
        
        # Extract images
        images = []
        if self.config.extract_images:
            images = self._extract_images(soup, url)
        
        # Validate content length
        if len(content) < self.config.min_content_length:
            logger.warning(f"Content too short for {url}: {len(content)} chars")
        elif len(content) > self.config.max_content_length:
            logger.warning(f"Content too long for {url}: {len(content)} chars")
            content = content[:self.config.max_content_length]
        
        return {
            'url': url,
            'success': True,
            'content': content,
            'metadata': metadata,
            'links': links,
            'images': images,
            'content_length': len(content),
            'scraped_at': time.time()
        }
    
    def _extract_metadata(self, soup: BeautifulSoup, response: requests.Response, url: str) -> Dict[str, Any]:
        """Extract metadata from page"""
        metadata = {
            'title': '',
            'description': '',
            'keywords': '',
            'author': '',
            'language': '',
            'content_type': response.headers.get('content-type', ''),
            'status_code': response.status_code,
            'url': url
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            
            if name == 'description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = content
            elif name == 'author':
                metadata['author'] = content
            elif name == 'language':
                metadata['language'] = content
        
        # Extract language from html tag
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')
        
        return metadata
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract links from page"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().strip()
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Validate URL
            if self._is_valid_url(absolute_url):
                links.append({
                    'url': absolute_url,
                    'text': text,
                    'title': link.get('title', '')
                })
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract images from page"""
        images = []
        
        for img in soup.find_all('img', src=True):
            src = img['src']
            alt = img.get('alt', '')
            title = img.get('title', '')
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, src)
            
            # Validate URL
            if self._is_valid_url(absolute_url):
                images.append({
                    'url': absolute_url,
                    'alt': alt,
                    'title': title
                })
        
        return images
    
    def _apply_content_filters(self, content: str) -> str:
        """Apply content filters"""
        for filter_pattern in self.config.content_filters:
            content = re.sub(filter_pattern, '', content, flags=re.IGNORECASE)
        
        return content
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if domain is allowed"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check blocked domains
            for blocked in self.config.blocked_domains:
                if blocked.lower() in domain:
                    return False
            
            # Check allowed domains (if specified)
            if self.config.allowed_domains:
                for allowed in self.config.allowed_domains:
                    if allowed.lower() in domain:
                        return True
                return False
            
            return True
            
        except:
            return False
    
    def _get_search_results(self, query: str, search_engine: str, max_results: int) -> List[Dict[str, str]]:
        """Get search results from search engine"""
        # This is a simplified implementation
        # In practice, you'd use proper search APIs
        
        if search_engine == "google":
            # Use Google Custom Search API or web scraping
            search_url = f"https://www.google.com/search?q={query}&num={max_results}"
            # Implementation would go here
            return []
        else:
            logger.warning(f"Search engine {search_engine} not implemented")
            return []
    
    def _save_content(self, content: Dict[str, Any], filename: str):
        """Save scraped content to file"""
        try:
            # Create output directory
            output_dir = Path("scraped_content")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename
            url_hash = hashlib.md5(content['url'].encode()).hexdigest()[:8]
            filename = f"{filename}_{url_hash}"
            
            if self.config.output_format == "json":
                file_path = output_dir / f"{filename}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
            
            elif self.config.output_format == "txt":
                file_path = output_dir / f"{filename}.txt"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content['content'])
            
            elif self.config.output_format == "markdown":
                file_path = output_dir / f"{filename}.md"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {content['metadata'].get('title', 'Untitled')}\n\n")
                    f.write(f"**URL:** {content['url']}\n\n")
                    f.write(content['content'])
            
            logger.info(f"Saved content to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save content: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_urls': len(self.content_cache),
            'total_content_length': sum(len(item.get('content', '')) for item in self.content_cache.values()),
            'average_content_length': sum(len(item.get('content', '')) for item in self.content_cache.values()) / max(len(self.content_cache), 1)
        }
    
    def clear_cache(self):
        """Clear content cache"""
        self.content_cache.clear()
        logger.info("Content cache cleared")