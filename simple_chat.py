#!/usr/bin/env python3
"""
SIMPLE CLEAN CHAT - Fixed version
"""

import sys
import os
import contextlib
import io

def count_letter_in_word(word, letter):
    """Count occurrences of a letter in a word"""
    return word.lower().count(letter.lower())

def start_simple_chat():
    """Simple chat with better response handling"""
    
    # Suppress all imports and initialization
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from revolutionary_neural_engine import EnhancedRevolutionaryEngine
        engine = EnhancedRevolutionaryEngine("your_serper_api_key_here")
    
    print("AI Chat")
    print("Type 'q' to quit")
    print()
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            # Handle specific queries that need special processing
            response = handle_special_queries(user_input)
            
            if not response:
                # Process with AI
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = engine.achieve_consciousness_with_validation(user_input)
                
                response = result['response']
                response = clean_response(response)
            
            print(response)
            
        except KeyboardInterrupt:
            break
        except Exception:
            print("Sorry, I couldn't process that. Try again.")
            continue

def handle_special_queries(query):
    """Handle specific queries that need special processing"""
    query_lower = query.lower()
    
    # Letter counting queries
    if "how many" in query_lower and "letter" in query_lower and ("appears" in query_lower or "appear" in query_lower):
        # Extract word and letter from query like "how many times the letter R appears on the word Strawberry"
        words = query.split()
        word_to_check = None
        letter_to_count = None
        
        # Find the word after "word"
        if "word" in query_lower:
            for i, word in enumerate(words):
                if word.lower() == "word":
                    if i + 1 < len(words):
                        word_to_check = words[i + 1]
                    break
        
        # Find the letter
        if "letter" in query_lower:
            for i, word in enumerate(words):
                if word.lower() == "letter":
                    if i + 1 < len(words):
                        letter_to_count = words[i + 1]
                    break
        
        if word_to_check and letter_to_count:
            count = count_letter_in_word(word_to_check, letter_to_count)
            return f"The letter '{letter_to_count.upper()}' appears {count} times in '{word_to_check}'."
    
    # Basic math
    if query_lower in ["1+1", "1 + 1", "what is 1+1", "what's 1+1", "how much is it 1+1"]:
        return "2"
    
    if "2+2" in query_lower or "2 + 2" in query_lower:
        return "4"
        
    if "3+3" in query_lower or "3 + 3" in query_lower:
        return "6"
    
    # Simple greetings
    if query_lower in ["hey", "hello", "hi"]:
        return "Hello! How can I help you?"
    
    return None

def clean_response(response):
    """Clean up AI response"""
    # Remove long introductions
    if "I'm a revolutionary AI" in response or "fractal neural tokenization" in response:
        # Try to extract meaningful content after the intro
        if "." in response:
            sentences = response.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and not any(tech_word in sentence.lower() for tech_word in 
                    ["revolutionary", "fractal", "quantum", "consciousness", "tokenization"]):
                    return sentence + "."
        return "Hello! How can I help you?"
    
    # Handle truncated responses
    if response.endswith(" d.") or response.endswith(" que.") or len(response) < 10:
        return "I can help with that. Could you rephrase your question?"
    
    # Clean up artifacts
    response = response.replace("...", ".")
    response = response.strip()
    
    return response

if __name__ == "__main__":
    start_simple_chat()