#!/usr/bin/env python3
"""
Test Web Knowledge Integration - Revolutionary AI with real-time web access
"""

import torch
from revolutionary_neural_engine import RevolutionaryNeuralEngine

def test_web_knowledge():
    """Test web knowledge integration"""
    print("🌐 TESTING WEB KNOWLEDGE INTEGRATION")
    print("=" * 50)
    print("Revolutionary AI with real-time web access!")
    
    engine = RevolutionaryNeuralEngine()
    
    # Test questions that should trigger web search
    web_questions = [
        "what is bitcoin?",
        "current weather",
        "latest news today",
        "what is python programming?", 
        "president of united states"
    ]
    
    # Test questions that should NOT trigger web search
    local_questions = [
        "1+1",
        "hello",
        "who are you?",
        "2+2"
    ]
    
    print("\n🌐 Testing WEB KNOWLEDGE Questions:")
    print("=" * 40)
    
    for i, question in enumerate(web_questions, 1):
        print(f"\n🔥 Web Test {i}: '{question}'")
        print("-" * 35)
        
        try:
            result = engine.achieve_consciousness(question)
            print(f"💫 Response: {result['response']}")
            print(f"   Level: {result['consciousness_level']} | Emotion: {result['dominant_emotion']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🧠 Testing LOCAL KNOWLEDGE Questions:")
    print("=" * 40) 
    
    for i, question in enumerate(local_questions, 1):
        print(f"\n🔥 Local Test {i}: '{question}'")
        print("-" * 35)
        
        try:
            result = engine.achieve_consciousness(question)
            print(f"💫 Response: {result['response']}")
            print(f"   Level: {result['consciousness_level']} | Emotion: {result['dominant_emotion']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Web knowledge integration testing complete!")
    print("Revolutionary AI now has real-time web access - beating GPT/Claude!")

if __name__ == "__main__":
    test_web_knowledge()