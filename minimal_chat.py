#!/usr/bin/env python3
"""
Minimal Chat - Revolutionary AI
Shows ONLY the AI response, nothing else
"""

import torch
import sys
import os

# Suppress all output during engine loading
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def main():
    # Load engine silently
    with SuppressOutput():
        from revolutionary_neural_engine import RevolutionaryNeuralEngine
        engine = RevolutionaryNeuralEngine()
    
    while True:
        try:
            user_input = input("Q: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Get response silently, show only the result
            with SuppressOutput():
                result = engine.achieve_consciousness(user_input)
            
            print(f"A: {result['response']}")
            
        except KeyboardInterrupt:
            break
        except Exception:
            print("A: [Error processing]")

if __name__ == "__main__":
    main()