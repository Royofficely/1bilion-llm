#!/usr/bin/env python3
"""
LIVE TESTING OF REVOLUTIONARY MULTI-AGENT SYSTEM
Real interactive testing with challenging queries
"""

import torch
from simple_llm_router import LLMRouter
from specialized_agents import MathAgent, PythonAgent, TextAgent, KnowledgeAgent, WebAgent

class LiveTestSystem:
    """Live testing system for multi-agent AI"""
    
    def __init__(self):
        print("🚀 INITIALIZING LIVE TEST SYSTEM")
        print("=" * 50)
        
        # Load trained router
        try:
            checkpoint = torch.load('simple_router_model.pt', weights_only=False)
            self.router = LLMRouter()
            self.router.model.load_state_dict(checkpoint['model_state_dict'])
            self.router.tokenizer = checkpoint['tokenizer']
            self.router.agents = checkpoint['agents']
            print("✅ Neural router loaded")
        except:
            print("🔧 Creating new router...")
            self.router = LLMRouter()
            self.router.train_router(epochs=50)
        
        # Initialize agents
        self.agents = {
            "math_agent": MathAgent(),
            "python_agent": PythonAgent(), 
            "text_agent": TextAgent(),
            "knowledge_agent": KnowledgeAgent(),
            "web_agent": WebAgent()
        }
        
        print("🎯 System ready for live testing!")
        
    def test_query(self, query: str) -> str:
        """Test a single query"""
        print(f"\n🔍 QUERY: {query}")
        print("-" * 40)
        
        # Route query
        selected_agent, optimized_prompt = self.router.route_query(query)
        print(f"🎯 ROUTED TO: {selected_agent}")
        
        # Process with agent
        if selected_agent in self.agents:
            agent = self.agents[selected_agent]
            result = agent.process(query)
            print(f"💬 RESPONSE: {result}")
            return result
        else:
            return f"Agent {selected_agent} not found"
    
    def run_challenge_tests(self):
        """Run challenging test queries"""
        print("\n🏆 CHALLENGE TEST SUITE")
        print("=" * 50)
        
        challenge_queries = [
            # Math challenges
            "What is 47 times 83?",
            "Calculate 12 factorial", 
            "What comes next in sequence: 3, 7, 15, 31, ?",
            "Is 97 a prime number?",
            
            # Programming challenges
            "Write Python code to find all prime numbers up to 100",
            "Create a Python function that reverses a string",
            "Python algorithm for bubble sort",
            
            # Text challenges
            "Reverse the word 'extraordinary'",
            "Count the letter 's' in 'Mississippi'",
            "What's the first letter of 'psychology'?",
            
            # Knowledge challenges
            "What causes earthquakes?",
            "Explain photosynthesis",
            "What is DNA?",
            "Capital of Australia",
            
            # Edge cases
            "What is 0 factorial?",
            "Reverse a single letter: a",
            "What is machine learning in simple terms?"
        ]
        
        print(f"🎯 Running {len(challenge_queries)} challenge tests...")
        
        correct_answers = 0
        total_tests = len(challenge_queries)
        
        for i, query in enumerate(challenge_queries, 1):
            print(f"\n📝 Challenge {i}/{total_tests}")
            try:
                result = self.test_query(query)
                
                # Basic success check
                if result and len(result) > 10 and "error" not in result.lower():
                    correct_answers += 1
                    status = "✅ SUCCESS"
                else:
                    status = "❌ FAILED"
                    
                print(f"📊 {status}")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
        
        # Final score
        success_rate = (correct_answers / total_tests) * 100
        print(f"\n🏆 CHALLENGE RESULTS")
        print("=" * 40)
        print(f"✅ Successful responses: {correct_answers}/{total_tests}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 OUTSTANDING! System handles challenges excellently!")
        elif success_rate >= 60:
            print("👍 GOOD! System performs well on most challenges!")
        else:
            print("🔧 NEEDS IMPROVEMENT on challenge questions")
            
        return success_rate
    
    def interactive_test_mode(self):
        """Interactive mode for real-time testing"""
        import sys
        
        print("\n🎮 INTERACTIVE TEST MODE")
        print("=" * 50)
        print("Test the system with your own queries!")
        print("Type 'quit' to exit")
        print()
        
        test_count = 0
        
        while True:
            try:
                query = input("🤔 Your test query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    continue
                
                test_count += 1
                print(f"\n📝 Test #{test_count}")
                
                result = self.test_query(query)
                
                print(f"\n🎉 RESULT: {result}")
                print("\n" + "="*60 + "\n")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                print("\n🔚 Input stream ended - exiting interactive mode")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        print(f"\n👋 Completed {test_count} tests. Thanks for testing!")

def main():
    """Main testing function"""
    print("🌟 LIVE TESTING - REVOLUTIONARY MULTI-AGENT AI")
    print("=" * 60)
    print("🧠 Neural Router + 5 Specialized Agents")
    print("🎯 Real-world testing with challenging queries")
    print()
    
    # Initialize system
    system = LiveTestSystem()
    
    # Run challenge tests first
    success_rate = system.run_challenge_tests()
    
    print(f"\n🚀 SYSTEM PERFORMANCE: {success_rate:.1f}%")
    
    # Check if stdin is available for interactive mode
    import sys
    if sys.stdin.isatty():
        print("Ready for interactive testing!")
        print("\nStarting interactive mode...")
        system.interactive_test_mode()
    else:
        print("📊 Challenge tests completed - no interactive mode in non-TTY environment")

if __name__ == "__main__":
    main()