#!/usr/bin/env python3
"""
Test GPT Killer functions directly
"""

from gpt_killer_final import get_gpt_killing_response

# Test cases
test_queries = [
    "hey",
    "tome now in bangkok?", 
    "whats the date today?",
    "bitcoin price",
    "1+1",
    "weather"
]

print("Testing GPT Killer functions:")
print("=" * 40)

for query in test_queries:
    print(f"Query: {query}")
    response = get_gpt_killing_response(query)
    print(f"Response: {response}")
    print("-" * 20)