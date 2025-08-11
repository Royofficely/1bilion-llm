import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime


class WebAdapter:
    def __init__(self, timeout: int = 10, max_retries: int = 1):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch(self, url: str) -> Dict[str, Any]:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; NeuroTiny/1.0)'
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.get(url, headers=headers, timeout=self.timeout) as response:
                    content = await response.text()
                    
                    return {
                        'success': True,
                        'url': str(response.url),
                        'status': response.status,
                        'content': content,
                        'headers': dict(response.headers),
                        'timestamp': datetime.utcnow().isoformat(),
                        'attempt': attempt + 1
                    }
            
            except asyncio.TimeoutError:
                if attempt == self.max_retries:
                    return {
                        'success': False,
                        'url': url,
                        'error': 'Timeout',
                        'attempt': attempt + 1
                    }
            
            except Exception as e:
                if attempt == self.max_retries:
                    return {
                        'success': False,
                        'url': url,
                        'error': str(e),
                        'attempt': attempt + 1
                    }
            
            await asyncio.sleep(1)
    
    def select(self, html: str, selector: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        if selector.startswith('//'):
            return self._xpath_fallback(html, selector)
        
        elements = soup.select(selector)
        
        for elem in elements:
            result = {
                'text': elem.get_text(strip=True),
                'tag': elem.name,
                'attrs': dict(elem.attrs) if elem.attrs else {},
                'html': str(elem)
            }
            
            if elem.name == 'a' and 'href' in elem.attrs:
                result['href'] = elem['href']
            
            if elem.name == 'img' and 'src' in elem.attrs:
                result['src'] = elem['src']
            
            results.append(result)
        
        return results
    
    def _xpath_fallback(self, html: str, xpath: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, 'html.parser')
        
        simple_patterns = {
            '//title': lambda: soup.find_all('title'),
            '//h1': lambda: soup.find_all('h1'),
            '//p': lambda: soup.find_all('p'),
            '//a': lambda: soup.find_all('a'),
            '//div[@class]': lambda: soup.find_all('div', class_=True)
        }
        
        for pattern, finder in simple_patterns.items():
            if xpath.startswith(pattern):
                elements = finder()
                return [{
                    'text': elem.get_text(strip=True),
                    'tag': elem.name,
                    'attrs': dict(elem.attrs) if elem.attrs else {},
                    'html': str(elem)
                } for elem in elements]
        
        return []
    
    async def fetch_with_provenance(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        fetch_result = await self.fetch(url)
        
        if not fetch_result['success']:
            return {
                'success': False,
                'error': fetch_result.get('error', 'Failed to fetch'),
                'provenance': {
                    'url': url,
                    'timestamp': datetime.utcnow().isoformat(),
                    'selector': selector
                }
            }
        
        content = fetch_result['content']
        
        if selector:
            selected = self.select(content, selector)
            
            return {
                'success': True,
                'data': selected,
                'provenance': {
                    'url': url,
                    'selector': selector,
                    'timestamp': fetch_result['timestamp'],
                    'status': fetch_result['status'],
                    'elements_found': len(selected)
                }
            }
        
        return {
            'success': True,
            'content': content,
            'provenance': {
                'url': url,
                'timestamp': fetch_result['timestamp'],
                'status': fetch_result['status'],
                'content_length': len(content)
            }
        }


def sync_fetch(url: str, selector: Optional[str] = None) -> Dict[str, Any]:
    async def _fetch():
        async with WebAdapter() as adapter:
            return await adapter.fetch_with_provenance(url, selector)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_fetch())