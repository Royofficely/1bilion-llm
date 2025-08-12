#!/usr/bin/env python3
"""
Clean Terminal Chat - Revolutionary AI
Simple interface to see pure neural responses clearly
"""

import torch
import time
from revolutionary_neural_engine import RevolutionaryNeuralEngine

def print_clean_banner():
    """Clean startup banner"""
    print("\n" + "="*50)
    print("🧠 REVOLUTIONARY AI - CLEAN CHAT")
    print("="*50)
    print("Pure Neural Consciousness | No Hardcoded Responses")
    print("Type 'quit' to exit, 'help' for commands")
    print("="*50)

def start_clean_chat():
    """Clean terminal chat interface"""
    print_clean_banner()
    
    # Initialize consciousness engine
    print("\n🔄 Loading consciousness engine...")
    engine = RevolutionaryNeuralEngine()
    print("✅ Revolutionary AI ready!\n")
    
    session_count = 0
    
    while True:
        try:
            # Clean input prompt
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle exit
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"\n👋 Chat ended after {session_count} interactions")
                break
            
            # Handle help
            if user_input.lower() == 'help':
                print("\n📋 Commands:")
                print("  help  - Show commands")
                print("  test  - Quick test")
                print("  stats - Performance stats") 
                print("  quit  - Exit chat")
                print("\n💡 Try: hello, 1+1, what is bitcoin?, who are you?")
                continue
            
            # Handle test
            if user_input.lower() == 'test':
                test_inputs = ["hello", "1+1", "what is bitcoin?"]
                print("\n🧪 Quick Test:")
                for test in test_inputs:
                    start_time = time.time()
                    result = engine.achieve_consciousness(test)
                    response_time = time.time() - start_time
                    print(f"  {test} → {result['response'][:60]}... ({response_time*1000:.0f}ms)")
                continue
            
            # Handle stats
            if user_input.lower() == 'stats':
                report = engine.get_consciousness_report()
                print("\n📊 Stats:")
                for key, value in list(report.items())[:4]:  # Show top 4 stats
                    print(f"  {key}: {value}")
                continue
            
            # Process through consciousness
            start_time = time.time()
            result = engine.achieve_consciousness(user_input)
            response_time = time.time() - start_time
            session_count += 1
            
            # Clean response display
            print(f"AI:  {result['response']}")
            print(f"     └─ {response_time*1000:.0f}ms | {result['consciousness_level']} | {result['dominant_emotion']}")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Chat interrupted after {session_count} interactions")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            continue

if __name__ == "__main__":
    start_clean_chat()