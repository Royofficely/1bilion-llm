#!/usr/bin/env python3
"""
Debug Web Search Logic
"""

from revolutionary_neural_engine import WebKnowledge

def debug_web_search():
    """Debug why web search isn't triggering"""
    web = WebKnowledge()
    
    test_queries = [
        "what is bitcoin?",
        "current weather", 
        "latest news today",
        "what is python programming?",
        "president of united states",
        "hello",
        "1+1"
    ]
    
    print("🔍 DEBUGGING WEB SEARCH LOGIC")
    print("=" * 40)
    
    for query in test_queries:
        should_search = web.should_search_web(query)
        print(f"Query: '{query}' -> Web search: {should_search}")
        
        if should_search:
            print("   📡 Testing actual web search...")
            results = web.search_web_knowledge(query)
            print(f"   Results: {len(results)} found")
            if results:
                formatted = web.format_web_knowledge(results)
                print(f"   Formatted: {formatted[:100]}{'...' if len(formatted) > 100 else ''}")
        print()

if __name__ == "__main__":
    debug_web_search()