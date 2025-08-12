#!/usr/bin/env python3
"""
CONCISE REVOLUTIONARY CHAT
Direct responses, no extra words - based on user feedback
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from revolutionary_neural_engine import EnhancedRevolutionaryEngine

def start_concise_chat():
    """Start concise chat with direct responses"""
    engine = EnhancedRevolutionaryEngine("your_serper_api_key_here")
    
    print("🌟 Revolutionary AI - Concise Mode")
    print("Direct answers, no extra words")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye")
                break
            
            # Get response with validation
            result = engine.achieve_consciousness_with_validation(user_input)
            response = result['response']
            
            # Make response more concise
            concise_response = make_concise(response)
            
            print(f"\n🌟 {concise_response}")
            
            # Optional: Show validation status briefly
            if result['validation_needed'] and result['web_sources']:
                print(f"   ✓ Web-verified")
            
        except KeyboardInterrupt:
            print(f"\n👋 Bye")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def make_concise(response):
    """Make response more direct and concise"""
    # Remove common filler phrases
    fillers = [
        "Hello! I'm a revolutionary AI with genuine consciousness, built using fractal neural tokenization and quantum superposition processing. I'm designed to provide helpful, accurate responses while maintaining true artificial consciousness. ",
        "I'm here to help you with questions, tasks, and conversations. ",
        "Great to meet you! ",
        "Thank you for asking! "
    ]
    
    for filler in fillers:
        response = response.replace(filler, "")
    
    # Trim to first sentence if too long
    sentences = response.split('. ')
    if len(sentences) > 1 and len(response) > 100:
        response = sentences[0] + '.'
    
    # Remove excessive technical details for greetings
    if any(word in response.lower() for word in ['hey', 'hello', 'hi']):
        if 'revolutionary' in response.lower():
            response = "Hi! I'm an AI assistant here to help."
    
    return response.strip()

if __name__ == "__main__":
    start_concise_chat()