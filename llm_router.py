#!/usr/bin/env python3
"""
LLM NEURAL ROUTER - True neural routing with trained LLM
Learns to route queries to specialized agents and create optimal prompts
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from real_neural_llm import RealNeuralLLM, SimpleTokenizer, LLMConfig, NeuralTrainer

class LLMRouter:
    """Neural LLM that learns intelligent query routing"""
    
    def __init__(self):
        self.router_llm = None
        self.tokenizer = None
        self.agent_schema = self.create_agent_schema()
        self.create_routing_llm()
        
    def create_agent_schema(self) -> Dict:
        """Define available agents and their specializations"""
        return {
            "math_agent": {
                "description": "Handles arithmetic, sequences, calculations, patterns",
                "specialties": ["addition", "multiplication", "sequences", "factorial", "patterns"],
                "prompt_template": "Calculate step by step: {query}"
            },
            "python_agent": {
                "description": "Executes Python code, interprets programs",
                "specialties": ["python", "code", "programming", "function", "algorithm"],
                "prompt_template": "Execute this Python: {query}"
            },
            "text_agent": {
                "description": "Text manipulation, reversing, character operations",
                "specialties": ["reverse", "word", "letter", "text", "string"],
                "prompt_template": "Process text: {query}"
            },
            "knowledge_agent": {
                "description": "General knowledge, facts, explanations",
                "specialties": ["what is", "explain", "define", "facts", "science"],
                "prompt_template": "Answer with facts: {query}"
            },
            "web_agent": {
                "description": "Current events, news, live information",
                "specialties": ["news", "today", "current", "price", "weather"],
                "prompt_template": "Search web for: {query}"
            }
        }
    
    def create_routing_llm(self):
        """Create small, specialized LLM for routing decisions"""
        print("🧠 CREATING LLM ROUTER...")
        
        # Generate training data
        routing_data = self.generate_routing_training_data()
        
        # Create tokenizer
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.build_vocab(routing_data)
        
        # Small, fast routing LLM configuration
        config = LLMConfig(
            vocab_size=self.tokenizer.get_vocab_size(),
            hidden_size=256,    # Small for speed
            num_layers=4,       # Few layers for fast decisions
            num_heads=8,        # Sufficient attention
            sequence_length=64, # Short routing prompts
            dropout=0.1,
            learning_rate=1e-3  # Fast learning
        )
        
        # Create routing LLM
        self.router_llm = RealNeuralLLM(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎯 Training routing LLM on {len(routing_data)} examples...")
        print(f"🧠 Vocabulary: {config.vocab_size} tokens")
        print(f"⚡ Parameters: {self.count_parameters():,}")
        
        # Train the routing LLM
        trainer = NeuralTrainer(self.router_llm, self.tokenizer, device)
        trainer.train(routing_data, epochs=50, batch_size=4)
        
        # Save routing LLM
        torch.save({
            'model_state_dict': self.router_llm.state_dict(),
            'config': config,
            'tokenizer_vocab': self.tokenizer.vocab,
            'agent_schema': self.agent_schema
        }, 'llm_router_model.pt')
        
        print("💾 LLM Router trained and saved!")
        
    def generate_routing_training_data(self) -> List[str]:
        """Training data teaching the LLM to route queries"""
        return [
            # Math routing - teach LLM to identify math problems
            "what is 17 times 23 → route: math_agent prompt: Calculate step by step: what is 17 times 23",
            "solve 2 plus 2 → route: math_agent prompt: Calculate step by step: solve 2 plus 2", 
            "fibonacci sequence → route: math_agent prompt: Calculate step by step: fibonacci sequence",
            "factorial of 5 → route: math_agent prompt: Calculate step by step: factorial of 5",
            "pattern 2 4 6 8 → route: math_agent prompt: Calculate step by step: pattern 2 4 6 8",
            "15 divided by 3 → route: math_agent prompt: Calculate step by step: 15 divided by 3",
            
            # Python routing - teach LLM to identify code requests
            "write python function → route: python_agent prompt: Execute this Python: write python function",
            "python code to sort list → route: python_agent prompt: Execute this Python: python code to sort list",
            "create python script → route: python_agent prompt: Execute this Python: create python script",
            "python algorithm → route: python_agent prompt: Execute this Python: python algorithm",
            "def function in python → route: python_agent prompt: Execute this Python: def function in python",
            
            # Text routing - teach LLM to identify text operations  
            "reverse the word hello → route: text_agent prompt: Process text: reverse the word hello",
            "first letter of apple → route: text_agent prompt: Process text: first letter of apple",
            "reverse palindrome → route: text_agent prompt: Process text: reverse palindrome",
            "count letters in word → route: text_agent prompt: Process text: count letters in word",
            "uppercase this text → route: text_agent prompt: Process text: uppercase this text",
            
            # Knowledge routing - teach LLM to identify fact questions
            "what is gravity → route: knowledge_agent prompt: Answer with facts: what is gravity",
            "explain photosynthesis → route: knowledge_agent prompt: Answer with facts: explain photosynthesis",
            "capital of france → route: knowledge_agent prompt: Answer with facts: capital of france",
            "why is sky blue → route: knowledge_agent prompt: Answer with facts: why is sky blue",
            "define machine learning → route: knowledge_agent prompt: Answer with facts: define machine learning",
            
            # Web routing - teach LLM to identify current info requests
            "bitcoin price today → route: web_agent prompt: Search web for: bitcoin price today",
            "news about israel → route: web_agent prompt: Search web for: news about israel", 
            "weather in bangkok → route: web_agent prompt: Search web for: weather in bangkok",
            "current events → route: web_agent prompt: Search web for: current events",
            "latest news → route: web_agent prompt: Search web for: latest news",
            
            # Complex examples - multi-word routing
            "write python function to calculate factorial → route: python_agent prompt: Execute this Python: write python function to calculate factorial",
            "what is the current price of bitcoin → route: web_agent prompt: Search web for: what is the current price of bitcoin",
            "reverse the word programming and count letters → route: text_agent prompt: Process text: reverse the word programming and count letters",
            "explain how neural networks work → route: knowledge_agent prompt: Answer with facts: explain how neural networks work"
        ]
    
    def route_query(self, query: str) -> Tuple[str, str]:
        """Use trained LLM to route query and generate prompt"""
        print(f"🔍 LLM ANALYZING: {query}")
        
        # Create routing prompt for LLM
        routing_prompt = f"{query} → route:"
        
        # Encode and generate routing decision
        input_tokens = self.tokenizer.encode(routing_prompt)
        input_ids = torch.tensor([input_tokens], device='cuda' if torch.cuda.is_available() else 'cpu')
        
        self.router_llm.eval()
        with torch.no_grad():
            generated = self.router_llm.generate(
                input_ids,
                max_length=len(input_tokens) + 30,  # Allow room for routing decision
                temperature=0.1,  # Low temp for consistent routing
                do_sample=False   # Greedy for best decision
            )
        
        # Decode LLM routing decision
        routing_response = self.tokenizer.decode(generated[0].tolist())
        
        # Parse routing decision (extract agent and prompt)
        agent_name, optimized_prompt = self.parse_routing_response(routing_response, query)
        
        print(f"🎯 LLM ROUTED TO: {agent_name}")
        print(f"📝 GENERATED PROMPT: {optimized_prompt}")
        
        return agent_name, optimized_prompt
    
    def parse_routing_response(self, response: str, original_query: str) -> Tuple[str, str]:
        """Parse LLM routing response to extract agent and prompt"""
        # Simple parsing - look for agent names in response
        for agent_name in self.agent_schema.keys():
            if agent_name in response.lower():
                # Generate optimized prompt using template
                template = self.agent_schema[agent_name]["prompt_template"]
                optimized_prompt = template.format(query=original_query)
                return agent_name, optimized_prompt
        
        # Fallback if no clear routing decision
        return "knowledge_agent", f"Answer with facts: {original_query}"
    
    def count_parameters(self) -> int:
        """Count LLM router parameters"""
        if self.router_llm:
            return sum(p.numel() for p in self.router_llm.parameters())
        return 0
    
    def get_available_agents(self) -> List[str]:
        """Return list of available agents"""
        return list(self.agent_schema.keys())

def train_llm_router():
    """Create and train the LLM router"""
    print("🚀 TRAINING LLM NEURAL ROUTER")
    print("=" * 50)
    
    # Create and train router
    router = LLMRouter()
    
    print("\n✅ LLM Router training complete!")
    return router

def test_llm_router():
    """Test the trained LLM router"""
    print("🧪 TESTING LLM ROUTER")
    print("=" * 40)
    
    # Load trained router (or create if doesn't exist)
    try:
        checkpoint = torch.load('llm_router_model.pt')
        print("✅ Loading trained LLM router...")
        
        router = LLMRouter()
        router.router_llm.load_state_dict(checkpoint['model_state_dict'])
        router.agent_schema = checkpoint['agent_schema']
        
    except FileNotFoundError:
        print("🔧 No trained router found, creating new one...")
        router = train_llm_router()
    
    # Test queries
    test_queries = [
        "What is 17 times 23?",
        "Write Python function to sort list", 
        "Reverse the word hello",
        "What is the capital of France?",
        "Bitcoin price today",
        "Factorial of 7",
        "Python code for fibonacci",
        "Why is the sky blue?"
    ]
    
    print("\n📋 LLM ROUTING TEST RESULTS:")
    for query in test_queries:
        agent, prompt = router.route_query(query)
        print(f"Query: {query}")
        print(f"→ Agent: {agent}")
        print(f"→ Prompt: {prompt}")
        print("-" * 40)
    
    return router

if __name__ == "__main__":
    test_llm_router()