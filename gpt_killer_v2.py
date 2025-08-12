#!/usr/bin/env python3
"""
GPT KILLER V2 - With Neural Router + Python Interpreter + Web Search
Revolutionary AI that beats GPT with multiple tools
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pattern_learning_engine import PatternLearningEngine, create_training_examples
from neural_router import NeuralRouter
from llm_processor import LLMProcessor
from gpt_killer_final import search_web

class GPTKillerV2:
    """GPT Killer with neural routing and multiple tools"""
    
    def __init__(self):
        # Initialize pattern learning engine
        self.pattern_engine = PatternLearningEngine()
        
        # Load training examples and train
        training_examples = create_training_examples()
        for input_text, output_text in training_examples:
            self.pattern_engine.add_example(input_text, output_text)
        self.pattern_engine.train()
        
        self.router = NeuralRouter()
        self.llm_processor = LLMProcessor()
        
        # Train the router with additional examples - send computation to pattern_learning
        training_data = [
            ("how many times letter r appears in strawberry", "pattern_learning"),
            ("count letter r in strawberry raspberry blueberry", "pattern_learning"),
            ("what is 15 + 25", "pattern_learning"),
            ("compute 100 / 4", "pattern_learning"),
            ("7 times 1.25", "pattern_learning"), 
            ("sarah has 3 brothers 2 sisters", "pattern_learning"),
            ("reverse palindrome", "pattern_learning"),
            ("who built you", "web_search"),
            ("current events", "web_search"),
            ("bitcoin price", "web_search"),
            ("thank you", "direct_answer"),
        ]
        self.router.train_routing(training_data)
    
    def get_response(self, query: str) -> str:
        """Get response using neural routing"""
        query_lower = query.lower().strip()
        
        # Route the query
        endpoint = self.router.route_query(query)
        
        # Execute based on routing decision
        if endpoint == "pattern_learning":
            return self.pattern_engine.predict(query)
        
        elif endpoint == "direct_answer":
            return self.get_direct_answer(query_lower)
        
        elif endpoint == "web_search":
            return self.get_web_answer(query, query_lower)
        
        else:
            # Fallback to web search
            return self.get_web_answer(query, query_lower)
    
    def get_direct_answer(self, query_lower: str) -> str:
        """Handle direct answers - NO HARDCODED RESPONSES"""
        # Even simple greetings go through web search
        return self.get_web_answer(query_lower, query_lower)
    
    def get_web_answer(self, query: str, query_lower: str) -> str:
        """Handle web search queries with intelligent LLM processing"""
        result = search_web(query)
        if result:
            # Process search results through LLM processor for intelligent responses
            return self.llm_processor.process_web_results(query, result)
        
        return "I couldn't find information about that. Let me try a different search approach."
    
    def simplify_answer(self, answer: str) -> str:
        """Simplify long answers to key information"""
        # Keep first sentence only
        if '.' in answer:
            answer = answer.split('.')[0]
        
        # Limit length
        return answer[:100] if len(answer) > 100 else answer
    
    def get_multiple_results(self, query: str, result: dict) -> str:
        """Get multiple results for queries like 'who is X' when there are multiple people"""
        if not ('who is' in query.lower() and 'organic' in result and len(result['organic']) > 1):
            return None
            
        # Check if multiple different people/entities
        snippets = []
        for i, item in enumerate(result['organic'][:3]):  # Show top 3
            snippet = item.get('snippet', '')
            title = item.get('title', '')
            if snippet:
                clean_snippet = snippet.split('.')[0][:80]
                snippets.append(f"{i+1}. {clean_snippet}")
        
        if len(snippets) > 1:
            return " | ".join(snippets)
        
        return None

def start_gpt_killer_v2():
    """Start GPT Killer V2 with neural routing"""
    
    print("GPT Killer V2 - Neural Routing + Python + Web")
    print("Type 'q' to quit")
    print()
    
    ai = GPTKillerV2()
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Goodbye!")
                break
            
            # Get response with neural routing
            response = ai.get_response(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    start_gpt_killer_v2()