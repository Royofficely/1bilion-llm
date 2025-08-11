from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class RouteDecision:
    path: str
    experts: List[str]
    budget: Dict[str, float]
    confidence: float


class Router:
    def __init__(self, fast_threshold: float = 0.8):
        self.fast_threshold = fast_threshold
        self.classifier = None
    
    def route(self, meta: Dict[str, Any]) -> RouteDecision:
        query = meta.get('query', '')
        wants_json = meta.get('wants_json', False)
        url = meta.get('url', '')
        selector = meta.get('selector', '')
        
        complexity_score = self._calculate_complexity(query, wants_json, url, selector)
        
        if complexity_score < self.fast_threshold and wants_json and not url:
            return RouteDecision(
                path='FAST_PATH',
                experts=['struct_mini'],
                budget={'time': 0.5, 'compute': 0.2},
                confidence=0.9
            )
        
        experts = []
        budget = {}
        
        if url or selector:
            experts.extend(['reason_mini', 'tool_adapter', 'struct_mini', 'verifier'])
            budget = {'time': 2.0, 'compute': 0.8}
        else:
            experts.extend(['reason_mini', 'struct_mini'])
            if wants_json:
                experts.append('verifier')
            budget = {'time': 1.0, 'compute': 0.5}
        
        return RouteDecision(
            path='SLOW_PATH',
            experts=experts,
            budget=budget,
            confidence=0.7 if url else 0.85
        )
    
    def _calculate_complexity(self, query: str, wants_json: bool, url: str, selector: str) -> float:
        score = 0.0
        
        if len(query) > 100:
            score += 0.3
        
        if url:
            score += 0.4
        
        if selector:
            score += 0.2
        
        if wants_json:
            score -= 0.1
        
        complex_keywords = ['multiple', 'analyze', 'compare', 'extract', 'transform']
        for keyword in complex_keywords:
            if keyword in query.lower():
                score += 0.2
        
        simple_keywords = ['get', 'fetch', 'show', 'display', 'return']
        for keyword in simple_keywords:
            if keyword in query.lower():
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def update_routing_policy(self, feedback: Dict[str, Any]):
        pass