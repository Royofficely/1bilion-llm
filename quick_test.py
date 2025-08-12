#!/usr/bin/env python3
"""
QUICK TEST - Simple way to test individual questions
"""

from pure_llm_decision_system import PureLLMInference
import sys

def quick_test(question):
    """Test a single question quickly"""
    print(f"🤖 Testing: {question}")
    print("-" * 50)
    
    try:
        llm = PureLLMInference()
        response = llm.process_query(question)
        print(f"✅ Success!")
        return response
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Test from command line argument or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is 10 times 20?"
    
    result = quick_test(question)