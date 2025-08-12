#!/usr/bin/env python3
"""
INTERACTIVE TEST - Test Pure LLM Decision System Yourself
Simple interface to ask questions and see LLM decisions
"""

from pure_llm_decision_system import PureLLMInference
import sys
import time

def main():
    """Interactive testing interface"""
    print("🤖 PURE LLM DECISION SYSTEM - INTERACTIVE TEST")
    print("=" * 60)
    print("Ask any question and see how the LLM makes decisions!")
    print("Type 'quit', 'exit', or 'q' to stop")
    print("Type 'help' for example questions")
    print()
    
    # Initialize Pure LLM
    try:
        print("Loading Pure LLM Decision System...")
        llm = PureLLMInference()
        print("✅ Pure LLM loaded and ready!")
    except Exception as e:
        print(f"❌ Failed to load Pure LLM: {e}")
        print("Make sure you've run the training first:")
        print("python3 pure_llm_decision_system.py")
        return
    
    print("\n🎯 Ready for your questions!")
    print("-" * 40)
    
    test_count = 0
    
    while True:
        try:
            # Get user input
            query = input("\n🤔 Your question: ").strip()
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() in ['help', 'h']:
                show_examples()
                continue
            elif not query:
                continue
            
            test_count += 1
            print(f"\n📝 Test #{test_count}")
            print("-" * 30)
            
            # Process with Pure LLM
            start_time = time.time()
            response = llm.process_query(query)
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            print(f"🎉 FINAL ANSWER: {response}")
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print(f"\n📊 You tested {test_count} questions!")
    print("Thanks for testing the Pure LLM Decision System! 🤖")

def show_examples():
    """Show example questions"""
    print("\n💡 EXAMPLE QUESTIONS TO TRY:")
    print("-" * 40)
    print("🔢 Math:")
    print("  • What is 123 times 456?")
    print("  • Find the 12th Fibonacci number")
    print("  • What is the derivative of x^2 + 3x?")
    print("  • Solve: 2x + 5 = 13")
    print("  • Is 89 a prime number?")
    
    print("\n📝 Text Processing:")
    print("  • Reverse the word 'hello'")
    print("  • Count the letter 'a' in 'banana'")
    print("  • What's the first letter of 'elephant'?")
    print("  • Check if 'race' and 'care' are anagrams")
    
    print("\n🧠 Knowledge:")
    print("  • What is DNA?")
    print("  • Capital of France")
    print("  • What causes rain?")
    print("  • Explain gravity")
    
    print("\n💻 Programming:")
    print("  • Write Python code to sort a list")
    print("  • Create a function to check even numbers")
    print("  • Python code for factorial")
    
    print("\n🧮 Complex:")
    print("  • If car travels 60mph for 2 hours, how far?")
    print("  • Calculate area of rectangle 5x8")
    print("  • What is 2^10?")

if __name__ == "__main__":
    main()