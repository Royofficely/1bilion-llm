from typing import Dict, Any, List, Optional
import json
import re
from datetime import datetime


class EvidenceNormalizer:
    def __init__(self):
        self.standard_fields = {
            'source_url': str,
            'timestamp': str,
            'selector': str,
            'content': str,
            'extracted_data': dict
        }
    
    def normalize(self, raw_evidence: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            'source_url': '',
            'timestamp': datetime.utcnow().isoformat(),
            'selector': '',
            'content': '',
            'extracted_data': {}
        }
        
        if 'provenance' in raw_evidence:
            prov = raw_evidence['provenance']
            normalized['source_url'] = prov.get('url', '')
            normalized['timestamp'] = prov.get('timestamp', normalized['timestamp'])
            normalized['selector'] = prov.get('selector', '')
        
        if 'url' in raw_evidence:
            normalized['source_url'] = raw_evidence['url']
        
        if 'data' in raw_evidence:
            normalized['extracted_data'] = self._extract_data(raw_evidence['data'])
        
        if 'content' in raw_evidence:
            normalized['content'] = str(raw_evidence['content'])[:1000]
        
        return normalized
    
    def _extract_data(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            return data
        
        if isinstance(data, list):
            extracted = {}
            
            for i, item in enumerate(data[:10]):
                if isinstance(item, dict):
                    if 'text' in item:
                        extracted[f'item_{i}'] = item['text']
                    elif 'value' in item:
                        extracted[f'item_{i}'] = item['value']
                    else:
                        extracted[f'item_{i}'] = str(item)
                else:
                    extracted[f'item_{i}'] = str(item)
            
            return extracted
        
        return {'value': str(data)}
    
    def merge_evidence(self, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not evidence_list:
            return {}
        
        merged = {
            'sources': [],
            'timestamps': [],
            'all_data': {}
        }
        
        for i, evidence in enumerate(evidence_list):
            normalized = self.normalize(evidence)
            
            if normalized['source_url']:
                merged['sources'].append(normalized['source_url'])
            
            merged['timestamps'].append(normalized['timestamp'])
            
            for key, value in normalized['extracted_data'].items():
                merged['all_data'][f'source_{i}_{key}'] = value
        
        merged['sources'] = list(set(merged['sources']))
        merged['earliest_timestamp'] = min(merged['timestamps']) if merged['timestamps'] else ''
        merged['latest_timestamp'] = max(merged['timestamps']) if merged['timestamps'] else ''
        
        return merged
    
    def validate_evidence(self, evidence: Dict[str, Any]) -> bool:
        required_fields = ['source_url', 'timestamp']
        
        for field in required_fields:
            if field not in evidence or not evidence[field]:
                return False
        
        try:
            datetime.fromisoformat(evidence['timestamp'].replace('Z', '+00:00'))
        except:
            return False
        
        return True


class EvidenceExtractor:
    def __init__(self):
        self.patterns = {
            'price': re.compile(r'[$£€]\s*(\d+(?:[.,]\d+)?)', re.I),
            'date': re.compile(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}'),
            'time': re.compile(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?', re.I),
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'phone': re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'url': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        }
    
    def extract_from_text(self, text: str) -> Dict[str, List[str]]:
        extracted = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                extracted[pattern_name] = matches[:5]
        
        return extracted
    
    def extract_structured_data(self, html_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        structured = {
            'titles': [],
            'links': [],
            'images': [],
            'text_content': []
        }
        
        for elem in html_elements:
            if elem.get('tag') in ['h1', 'h2', 'h3', 'title']:
                structured['titles'].append(elem.get('text', ''))
            
            if elem.get('href'):
                structured['links'].append({
                    'text': elem.get('text', ''),
                    'href': elem.get('href')
                })
            
            if elem.get('src'):
                structured['images'].append({
                    'alt': elem.get('attrs', {}).get('alt', ''),
                    'src': elem.get('src')
                })
            
            if elem.get('text'):
                structured['text_content'].append(elem.get('text'))
        
        for key in list(structured.keys()):
            if not structured[key]:
                del structured[key]
        
        return structured