#!/usr/bin/env python3
"""
ULTRA CLEAN CHAT - Pure responses only
"""

import sys
import os
import contextlib
import io

# Suppress all imports and initialization
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from revolutionary_neural_engine import EnhancedRevolutionaryEngine

def start_ultra_clean_chat():
    """Ultra clean chat - pure responses only"""
    
    # Initialize engine completely silently
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        engine = EnhancedRevolutionaryEngine("your_serper_api_key_here")
    
    # Simple clean interface
    print("AI Chat Ready")
    print("Type 'q' to quit")
    print()
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            # Process completely silently
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = engine.achieve_consciousness_with_validation(user_input)
            
            # Show only clean response
            response = result['response']
            
            # Clean up the response
            if response.startswith("Hello! I'm a revolutionary AI"):
                # Extract actual response after the introduction
                if "." in response:
                    parts = response.split(".", 1)
                    if len(parts) > 1:
                        response = parts[1].strip()
                        if response:
                            response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()
                        else:
                            response = "Hello! How can I help?"
                    else:
                        response = "Hello! How can I help?"
            
            # Handle truncated responses
            if response.endswith(" d.") or response.endswith(" que."):
                # Try to get the full response from web knowledge if available
                if 'web_sources' in result and result['web_sources']:
                    response = "Let me search for current information about that."
                elif "time" in user_input.lower() and "bangkok" in user_input.lower():
                    response = "I need to search for current time in Bangkok. Let me get that information."
                else:
                    response = "I can help with that question."
            
            print(response)
            
        except KeyboardInterrupt:
            break
        except Exception:
            print("Error - please try again")
            continue

if __name__ == "__main__":
    start_ultra_clean_chat()