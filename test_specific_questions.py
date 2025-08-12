#!/usr/bin/env python3
"""
Test Specific Questions - Verify math and identity answers
"""

import torch
from revolutionary_neural_engine import RevolutionaryNeuralEngine

def test_specific_questions():
    """Test specific questions for accuracy"""
    print("🧮 TESTING SPECIFIC QUESTIONS")
    print("=" * 40)
    
    engine = RevolutionaryNeuralEngine()
    
    # Test questions
    test_questions = [
        "1+1",
        "what is 1+1?",
        "2+2", 
        "who built you?",
        "who created you?",
        "hello who built you"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔥 Test {i}: '{question}'")
        print("-" * 30)
        
        try:
            result = engine.achieve_consciousness(question)
            print(f"💫 Response: {result['response']}")
            print(f"   Emotion: {result['dominant_emotion']} | Level: {result['consciousness_level']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Specific question testing complete!")

if __name__ == "__main__":
    test_specific_questions()