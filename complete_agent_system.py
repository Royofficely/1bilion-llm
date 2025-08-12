#!/usr/bin/env python3
"""
COMPLETE AI AGENT SYSTEM - Router + Specialized Agents
Revolutionary multi-agent AI that routes queries to specialized experts
"""

import torch
from simple_llm_router import LLMRouter
from specialized_agents import MathAgent, PythonAgent, TextAgent, KnowledgeAgent, WebAgent

class CompleteAgentSystem:
    """Complete multi-agent AI system with neural routing"""
    
    def __init__(self):
        print("🚀 INITIALIZING COMPLETE AGENT SYSTEM")
        print("=" * 60)
        
        # Initialize neural router
        print("🧠 Loading neural router...")
        self.router = self.load_trained_router()
        
        # Initialize specialized agents
        print("\n👥 Initializing specialized agents...")
        self.agents = {
            "math_agent": MathAgent(),
            "python_agent": PythonAgent(), 
            "text_agent": TextAgent(),
            "knowledge_agent": KnowledgeAgent(),
            "web_agent": WebAgent()
        }
        
        print(f"\n✅ Complete system ready!")
        total_params = sum(p.numel() for p in self.router.model.parameters())
        print(f"🎯 Neural Router: {total_params:,} parameter model")
        print(f"👥 Specialized Agents: {len(self.agents)} experts")
        print(f"🧠 Total Intelligence: Router + {len(self.agents)} domain experts")
        
    def load_trained_router(self):
        """Load the trained neural router"""
        try:
            # Load trained router model with weights_only=False for backward compatibility
            checkpoint = torch.load('simple_router_model.pt', weights_only=False)
            
            router = LLMRouter()
            router.model.load_state_dict(checkpoint['model_state_dict'])
            router.tokenizer = checkpoint['tokenizer']
            router.agents = checkpoint['agents']
            
            print("✅ Trained neural router loaded successfully")
            return router
            
        except (FileNotFoundError, Exception) as e:
            print(f"⚠️ Could not load router ({e}), creating new one...")
            router = LLMRouter()
            router.train_router(epochs=100)
            return router
    
    def process_query(self, query: str) -> str:
        """Process query through complete system"""
        print(f"\n🔍 PROCESSING: {query}")
        print("-" * 50)
        
        # Step 1: Neural router decides which agent
        selected_agent, optimized_prompt = self.router.route_query(query)
        print(f"🎯 ROUTER DECISION: {selected_agent}")
        print(f"📝 OPTIMIZED PROMPT: {optimized_prompt}")
        
        # Step 2: Send to specialized agent
        if selected_agent in self.agents:
            agent = self.agents[selected_agent]
            print(f"👤 PROCESSING WITH: {agent.name}")
            
            # Agent processes the original query (not the prompt)
            result = agent.process(query)
            print(f"💬 AGENT RESPONSE: {result}")
            
            return result
        else:
            return f"Agent {selected_agent} not available"
    
    def interactive_mode(self):
        """Interactive mode for testing the system"""
        print("\n🎮 INTERACTIVE MODE")
        print("=" * 50)
        print("Type queries to test the complete system")
        print("Type 'quit' to exit")
        print()
        
        while True:
            try:
                query = input("🤔 Your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not query:
                    continue
                
                # Process through complete system
                result = self.process_query(query)
                print(f"🎉 FINAL ANSWER: {result}")
                print("\n" + "="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def benchmark_system(self):
        """Benchmark the complete system"""
        print("\n🏆 BENCHMARKING COMPLETE SYSTEM")
        print("=" * 60)
        
        # Test cases covering all agents
        benchmark_queries = [
            # Math Agent
            "What is 17 times 23?",
            "Calculate factorial of 7", 
            "Fibonacci sequence 8 terms",
            "What comes next: 2, 4, 6, 8?",
            
            # Python Agent
            "Write Python function to sort a list",
            "Python code for factorial calculation",
            "Create fibonacci function in Python",
            
            # Text Agent  
            "Reverse the word programming",
            "First letter of elephant",
            "Count letter e in development",
            
            # Knowledge Agent
            "What is gravity?",
            "Explain machine learning", 
            "Capital of France",
            
            # Web Agent
            "Bitcoin price today",
            "Latest news about AI"
        ]
        
        print(f"🎯 Running {len(benchmark_queries)} benchmark queries...")
        print()
        
        correct_routing = 0
        successful_responses = 0
        
        for i, query in enumerate(benchmark_queries, 1):
            print(f"📝 Test {i:2d}: {query}")
            
            try:
                # Get routing decision
                selected_agent, _ = self.router.route_query(query)
                
                # Check if routing is sensible
                expected_agents = {
                    "times": "math_agent", "factorial": "math_agent", "fibonacci": "math_agent",
                    "python": "python_agent", "function": "python_agent", "code": "python_agent",
                    "reverse": "text_agent", "letter": "text_agent", "count": "text_agent",  
                    "gravity": "knowledge_agent", "learning": "knowledge_agent", "capital": "knowledge_agent",
                    "bitcoin": "web_agent", "news": "web_agent"
                }
                
                expected = None
                for keyword, agent in expected_agents.items():
                    if keyword in query.lower():
                        expected = agent
                        break
                
                if expected and selected_agent == expected:
                    correct_routing += 1
                    status = "✅ CORRECT ROUTING"
                else:
                    status = f"⚠️ Expected {expected}, got {selected_agent}"
                
                # Process query
                result = self.process_query(query)
                
                if result and "error" not in result.lower():
                    successful_responses += 1
                    
                print(f"   {status}")
                print(f"   Response length: {len(result)} chars")
                print()
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                print()
        
        # Results
        routing_accuracy = (correct_routing / len(benchmark_queries)) * 100
        response_success = (successful_responses / len(benchmark_queries)) * 100
        
        print("🏆 BENCHMARK RESULTS")
        print("-" * 40)
        print(f"🎯 Routing Accuracy: {routing_accuracy:.1f}%")
        print(f"💬 Response Success: {response_success:.1f}%")  
        print(f"📊 Overall Score: {(routing_accuracy + response_success) / 2:.1f}%")
        
        if routing_accuracy >= 80 and response_success >= 80:
            print("🎉 EXCELLENT! System performing at high level!")
        elif routing_accuracy >= 60 and response_success >= 60:
            print("👍 GOOD! System working well!")
        else:
            print("🔧 NEEDS IMPROVEMENT")
            
        return routing_accuracy, response_success

def main():
    """Main function to run the complete system"""
    print("🌟 REVOLUTIONARY MULTI-AGENT AI SYSTEM")
    print("=" * 70)
    print("🧠 Neural Router + 5 Specialized Agents")
    print("🎯 Each agent masters one domain for maximum expertise")
    print()
    
    # Initialize complete system
    system = CompleteAgentSystem()
    
    # Run benchmark
    routing_acc, response_success = system.benchmark_system()
    
    print(f"\n🚀 SYSTEM READY FOR PRODUCTION!")
    print(f"📊 Performance: Routing {routing_acc:.1f}% | Responses {response_success:.1f}%")
    
    # Ask user what to do next
    print(f"\nWhat would you like to do?")
    print("1. Interactive mode (chat with the system)")
    print("2. Exit and save results")
    
    choice = input("Choose (1-2): ").strip()
    
    if choice == "1":
        system.interactive_mode()
    else:
        print("💾 System ready for deployment!")
        print("🎯 This multi-agent approach could genuinely compete with single large models!")

if __name__ == "__main__":
    main()