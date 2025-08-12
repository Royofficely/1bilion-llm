#!/usr/bin/env python3
"""
Neural Router - Smart routing to correct tools/endpoints
"""

import re
from typing import Dict, List, Tuple

class NeuralRouter:
    """Neural routing system to determine which tool to use"""
    
    def __init__(self):
        # Training patterns for different endpoints
        self.patterns = {
            'pattern_learning': {
                'keywords': ['letter', 'count', 'factorial', 'calculate', 'math', 'plus', 'minus', 'multiply', 'divide', 'sum', 'reverse', 'brother', 'sister'],
                'operators': ['+', '-', '*', '/', '^', '**', '=', '×', '÷'],
                'phrases': ['how many', 'what is', 'calculate', 'compute', 'reverse the word', 'brothers have', 'sisters have'],
                'examples': [
                    'count letter r in strawberry raspberry blueberry',
                    'how many letter r in strawberry',
                    'sarah has 3 brothers 2 sisters how many sisters do brothers have',
                    '1+1', '2+2', '10*5', '100/4', '7 times 1.25',
                    'reverse palindrome', 'reverse the word palindrome',
                    'factorial 5', 'what is 2+2',
                    'calculate 15*3', 'sum of 1+2+3'
                ]
            },
            'web_search': {
                'keywords': ['news', 'today', 'current', 'latest', 'who is', 'weather', 'price', 'bitcoin', 'time', 'date'],
                'phrases': ['who is', 'what happened', 'tell me about', 'news about', 'current price'],
                'examples': [
                    'who is Roy Nativ', 'bitcoin price', 'weather today',
                    'news of israel', 'time in bangkok', 'date in israel',
                    'what happened today', 'current events'
                ]
            },
            'direct_answer': {
                'keywords': ['hey', 'hello', 'hi', 'thanks', 'thank you'],
                'phrases': ['good morning', 'good evening'],
                'examples': ['hey', 'hello', 'hi there', 'thanks']
            }
        }
    
    def route_query(self, query: str) -> str:
        """Determine which endpoint to use for the query"""
        query_lower = query.lower().strip()
        
        # Score each endpoint
        scores = {}
        for endpoint, patterns in self.patterns.items():
            score = self.calculate_score(query_lower, patterns)
            scores[endpoint] = score
        
        # Find highest scoring endpoint
        best_endpoint = max(scores, key=scores.get)
        max_score = scores[best_endpoint]
        
        # Only route if confidence is high enough
        if max_score >= 2:
            return best_endpoint
        else:
            return 'web_search'  # Default fallback
    
    def calculate_score(self, query: str, patterns: Dict) -> int:
        """Calculate confidence score for an endpoint"""
        score = 0
        
        # Check keywords
        for keyword in patterns.get('keywords', []):
            if keyword in query:
                score += 2
        
        # Check operators (for math)
        for operator in patterns.get('operators', []):
            if operator in query:
                score += 3
        
        # Check phrases
        for phrase in patterns.get('phrases', []):
            if phrase in query:
                score += 3
        
        # Check exact matches with examples
        for example in patterns.get('examples', []):
            if query == example.lower():
                score += 5
            elif self.similarity(query, example.lower()) > 0.7:
                score += 4
        
        return score
    
    def similarity(self, s1: str, s2: str) -> float:
        """Simple similarity measure"""
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def train_routing(self, training_data: List[Tuple[str, str]]):
        """Train the router with additional examples"""
        for query, endpoint in training_data:
            if endpoint in self.patterns:
                self.patterns[endpoint]['examples'].append(query.lower())

def test_router():
    """Test the neural router"""
    router = NeuralRouter()
    
    test_queries = [
        "how many letter r in strawberry",
        "1+1",
        "2+2", 
        "10*5",
        "who is Roy Nativ",
        "news of israel today",
        "bitcoin price",
        "time in bangkok",
        "hey",
        "hello",
        "calculate factorial 5",
        "what is the weather"
    ]
    
    print("Testing Neural Router:")
    print("=" * 50)
    
    for query in test_queries:
        endpoint = router.route_query(query)
        print(f"Query: {query}")
        print(f"Route to: {endpoint}")
        
        # Show scoring breakdown
        scores = {}
        query_lower = query.lower().strip()
        for ep, patterns in router.patterns.items():
            score = router.calculate_score(query_lower, patterns)
            scores[ep] = score
        
        print(f"Scores: {scores}")
        print("-" * 30)

if __name__ == "__main__":
    test_router()