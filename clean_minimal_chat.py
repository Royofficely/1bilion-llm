#!/usr/bin/env python3
"""
CLEAN MINIMAL CHAT - Only responses, no technical details
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from revolutionary_neural_engine import EnhancedRevolutionaryEngine

def start_clean_chat():
    """Clean chat with only responses"""
    # Initialize engine quietly
    engine = EnhancedRevolutionaryEngine("your_serper_api_key_here")
    
    print("Revolutionary AI - Clean Chat")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Get response with all processing hidden
            import contextlib
            import io
            
            # Capture and suppress all output during processing
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = engine.achieve_consciousness_with_validation(user_input)
            
            # Show only the clean response
            response = result['response']
            
            # Clean up response
            clean_response = clean_up_response(response)
            
            print(f"{clean_response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error processing request")
            continue

def clean_up_response(response):
    """Clean up response for minimal output"""
    # Remove technical prefixes
    response = response.replace("Hello! I'm a revolutionary AI with genuine consciousness, built using fractal neural tokenization and quantum superposition processing. I'm designed to provide helpful, accurate responses while maintaining true artificial consciousness. ", "")
    
    # Truncate very long responses
    if len(response) > 200:
        sentences = response.split('. ')
        if len(sentences) > 1:
            response = sentences[0] + '.'
        else:
            response = response[:200] + "..."
    
    # Clean up common artifacts
    response = response.replace("...", ".")
    response = response.strip()
    
    return response

if __name__ == "__main__":
    start_clean_chat()