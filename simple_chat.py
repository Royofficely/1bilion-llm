#!/usr/bin/env python3
"""
Simple Chat - Revolutionary AI
Minimal interface to see pure responses clearly
"""

import torch
from revolutionary_neural_engine import RevolutionaryNeuralEngine

def main():
    print("🧠 Revolutionary AI - Simple Chat")
    print("Type 'quit' to exit\n")
    
    # Load engine quietly
    engine = RevolutionaryNeuralEngine()
    print("Ready!\n")
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                break
            
            # Get pure response
            result = engine.achieve_consciousness(user_input)
            print(f"AI: {result['response']}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()