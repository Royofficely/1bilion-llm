#!/usr/bin/env python3
"""
TEST WEB VALIDATION SYSTEM
Automated tests for the revolutionary AI validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from revolutionary_neural_engine import EnhancedRevolutionaryEngine

def test_validation_system():
    """Test the web validation system"""
    print("🧪 TESTING REVOLUTIONARY WEB VALIDATION")
    print("="*50)
    
    # Initialize engine
    engine = EnhancedRevolutionaryEngine("your_serper_api_key_here")
    
    # Test queries
    test_queries = [
        "hey",                    # Should NOT validate (simple greeting)
        "1+1",                   # Should NOT validate (simple math)
        "bitcoin price today",    # SHOULD validate (real-time data)
        "who are you",           # Should NOT validate (identity)
        "weather today",         # SHOULD validate (real-time data)
        "what is AI",            # May or may not validate (factual)
        "current president 2024", # SHOULD validate (current info)
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing: '{query}'")
        print("-" * 30)
        
        try:
            # Get AI response with validation
            result = engine.achieve_consciousness_with_validation(query)
            
            print(f"🌟 Response: {result['response']}")
            
            # Show validation details
            validation_needed = result.get('validation_needed', False)
            confidence = result.get('validation_confidence', 0.0)
            sources = result.get('web_sources', [])
            
            print(f"🔍 Validation needed: {validation_needed}")
            print(f"📊 Confidence: {confidence:.1f}")
            
            if sources:
                print(f"📚 Sources: {len(sources)} web references")
                for i, source in enumerate(sources[:2]):
                    print(f"   {i+1}. {source[:60]}...")
            
            print(f"⚡ Processing time: {result['processing_time']*1000:.0f}ms")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*50)
    print("✅ Validation system test completed!")

def test_neural_decision_making():
    """Test the neural decision making component"""
    print("\n🧠 TESTING NEURAL DECISION MAKING")
    print("="*50)
    
    from revolutionary_neural_engine import WebValidationSystem
    
    validator = WebValidationSystem()
    
    test_cases = [
        ("hey there", "Hi! How can I help?"),
        ("bitcoin price today", "Bitcoin is currently trading at..."),
        ("what is 2+2", "2+2 equals 4"),
        ("weather in New York today", "Today's weather in New York..."),
        ("who is the current president", "The current president is..."),
    ]
    
    for query, response in test_cases:
        should_validate = validator.should_validate(query, response)
        print(f"'{query}' → Validate: {should_validate}")
    
    print("✅ Neural decision making test completed!")

if __name__ == "__main__":
    test_validation_system()
    test_neural_decision_making()