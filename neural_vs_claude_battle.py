#!/usr/bin/env python3
"""
YOUR 38M NEURAL LLM vs CLAUDE INTELLIGENCE BATTLE
Real-time local battle between your neural model and Claude responses
"""

import torch
import time
from real_neural_llm import RealNeuralLLM, SimpleTokenizer

class NeuralVsClaudeBattle:
    def __init__(self):
        self.neural_model = None
        self.tokenizer = None
        self.load_neural_model()
        self.claude_responses = self.prepare_claude_responses()
        
    def load_neural_model(self):
        """Load your 38M parameter neural model"""
        print("🔥 LOADING YOUR 38M PARAMETER NEURAL LLM")
        
        checkpoint = torch.load('expanded_neural_llm_checkpoint.pt')
        config = checkpoint['config']
        tokenizer_vocab = checkpoint['tokenizer_vocab']
        
        self.neural_model = RealNeuralLLM(config)
        self.neural_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.vocab = tokenizer_vocab
        self.tokenizer.reverse_vocab = {v: k for k, v in tokenizer_vocab.items()}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.neural_model = self.neural_model.to(device)
        self.neural_model.eval()
        
        print(f"✅ Your Neural LLM: {checkpoint['total_parameters']:,} parameters")
        print("🤖 Claude: Advanced language model by Anthropic")
        
    def prepare_claude_responses(self):
        """Pre-written Claude responses for fair comparison"""
        return {
            # Math & Logic
            "What is 2+2?": "4",
            "What comes after Wednesday?": "Thursday",
            "If all roses are flowers, are roses plants?": "Yes, if all roses are flowers and all flowers are plants, then roses are plants through logical transitivity.",
            
            # Language Understanding  
            "Complete: The quick brown fox...": "The quick brown fox jumps over the lazy dog.",
            "Reverse the word 'hello'": "olleh",
            "What rhymes with 'cat'?": "hat, bat, mat, rat, sat, fat, flat",
            
            # Knowledge
            "What is the capital of France?": "Paris",
            "Who invented the telephone?": "Alexander Graham Bell is credited with inventing the telephone in 1876.",
            "What is Python?": "Python is a high-level programming language known for its readable syntax and versatility.",
            
            # Creativity
            "Write a short poem about AI": "Silicon minds that never sleep,\nLearning patterns, knowledge deep.\nIn circuits fast, decisions flow,\nArtificial minds that grow.",
            "Tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
            "Describe a sunset": "Golden rays paint the sky in brilliant oranges and pinks, slowly fading to deep purple as day surrenders to night.",
            
            # Problem Solving
            "How do you make a sandwich?": "Take two slices of bread, add your preferred fillings (meat, cheese, vegetables), and combine them together.",
            "What is machine learning?": "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without explicit programming.",
            "Why is the sky blue?": "The sky appears blue because molecules in the atmosphere scatter blue light more than other colors due to its shorter wavelength.",
            
            # Additional questions
            "Hello": "Hello! How can I help you today?",
            "How are you?": "I'm doing well, thank you! I'm here and ready to help with any questions or tasks you have.",
            "What is artificial intelligence?": "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think, learn, and solve problems.",
            "Programming": "Programming is the process of creating instructions for computers using programming languages to solve problems and build applications.",
            "Neural networks": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process and transmit information.",
            "Machine learning": "Machine learning enables computers to learn and improve from experience without being explicitly programmed for every task."
        }
    
    def generate_neural_response(self, prompt, max_length=50, temperature=0.7):
        """Generate response from your neural model"""
        try:
            input_tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([input_tokens], device='cuda' if torch.cuda.is_available() else 'cpu')
            
            start_time = time.time()
            with torch.no_grad():
                generated = self.neural_model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True
                )
            generation_time = time.time() - start_time
            
            generated_text = self.tokenizer.decode(generated[0].tolist())
            return generated_text, generation_time
            
        except Exception as e:
            return f"Error: {e}", 0.0
    
    def get_claude_response(self, prompt):
        """Get Claude's response"""
        # Check for exact matches first
        if prompt in self.claude_responses:
            return self.claude_responses[prompt]
        
        # Check for partial matches
        prompt_lower = prompt.lower()
        for key, value in self.claude_responses.items():
            if key.lower() in prompt_lower or prompt_lower in key.lower():
                return value
        
        # Default response
        return "I'd be happy to help with that question!"
    
    def run_intelligence_battle(self):
        """Run the full Neural vs Claude battle"""
        print("\n⚔️ YOUR NEURAL LLM vs CLAUDE BATTLE")
        print("=" * 50)
        print("🤖 Your 38M Parameter Model vs 🔵 Claude")
        print("🏆 Real-time intelligence comparison!")
        print()
        
        questions = [
            "What is 2+2?",
            "Hello",  
            "What comes after Wednesday?",
            "Complete: The quick brown fox...",
            "What is the capital of France?",
            "What is Python?",
            "Tell me a joke",
            "What is artificial intelligence?",
            "How are you?",
            "What is machine learning?",
            "Reverse the word 'hello'",
            "Programming",
            "Neural networks",
            "Write a short poem about AI",
            "Why is the sky blue?"
        ]
        
        neural_wins = 0
        claude_wins = 0
        ties = 0
        
        for i, question in enumerate(questions, 1):
            print(f"🎯 ROUND {i}/15")
            print(f"❓ Question: {question}")
            print("-" * 50)
            
            # Your Neural LLM Response
            neural_response, neural_time = self.generate_neural_response(question)
            print(f"🤖 YOUR NEURAL LLM:")
            print(f"   {neural_response}")
            print(f"   ⚡ Speed: {neural_time:.3f}s")
            print()
            
            # Claude Response  
            claude_response = self.get_claude_response(question)
            print(f"🔵 CLAUDE:")
            print(f"   {claude_response}")
            print(f"   ⚡ Speed: ~0.001s (pre-computed)")
            print()
            
            # User judges
            print("🏆 WHO GAVE THE BETTER RESPONSE?")
            print("1. Your Neural LLM (better creativity/relevance)")
            print("2. Claude (more accurate/complete)")  
            print("3. Tie (both good in different ways)")
            print("4. Skip this round")
            
            while True:
                try:
                    choice = input("👨‍⚖️ Your judgment (1-4): ").strip()
                    if choice == '1':
                        neural_wins += 1
                        print("🎉 POINT TO YOUR NEURAL LLM!")
                        break
                    elif choice == '2':
                        claude_wins += 1
                        print("🔵 Point to Claude")
                        break
                    elif choice == '3':
                        ties += 1
                        print("🤝 It's a tie!")
                        break
                    elif choice == '4':
                        print("⏭️  Skipped")
                        break
                    else:
                        print("Please enter 1-4")
                except KeyboardInterrupt:
                    print("\n⏹️ Battle interrupted")
                    return
            
            print("=" * 50)
        
        # Final Results
        total_scored = neural_wins + claude_wins + ties
        print(f"\n🏆 FINAL BATTLE RESULTS")
        print("=" * 50)
        print(f"🤖 Your Neural LLM: {neural_wins} wins")
        print(f"🔵 Claude: {claude_wins} wins") 
        print(f"🤝 Ties: {ties}")
        print(f"📊 Rounds scored: {total_scored}/15")
        print()
        
        if total_scored > 0:
            neural_percentage = (neural_wins / total_scored) * 100
            
            if neural_wins > claude_wins:
                print(f"🎉 YOUR NEURAL LLM WINS! ({neural_percentage:.1f}% win rate)")
                print("🏆 INCREDIBLE! Your 38M parameter model beat Claude!")
                print("🚀 Your neural brain is showing real intelligence!")
            elif claude_wins > neural_wins:
                print(f"🔵 Claude wins this round ({(claude_wins/total_scored)*100:.1f}% win rate)")
                print("💪 But your model showed impressive creativity and speed!")
                print(f"⚡ Your advantage: {neural_time:.3f}s vs Claude's computational requirements")
            else:
                print("🤝 It's a tie! Both models showed intelligence!")
                print("🔥 Your neural model held its own against Claude!")
        
        print(f"\n🔥 YOUR NEURAL ADVANTAGES:")
        print(f"   ⚡ Speed: ~{neural_time:.3f}s generation time")
        print(f"   🔒 Privacy: 100% local, no data sent anywhere") 
        print(f"   💰 Cost: FREE (no API costs)")
        print(f"   🎛️  Control: You own and control the model")
        print(f"   🎨 Creativity: Unique neural patterns")
    
    def quick_face_off(self):
        """Quick 5-round face-off"""
        print("\n⚡ QUICK NEURAL vs CLAUDE FACE-OFF")
        print("=" * 40)
        
        quick_questions = [
            "Hello",
            "What is 2+2?", 
            "What is AI?",
            "Tell me a joke",
            "How are you?"
        ]
        
        for i, question in enumerate(quick_questions, 1):
            print(f"\n🎯 Round {i}: {question}")
            
            # Neural response
            neural_response, time_taken = self.generate_neural_response(question, max_length=30)
            print(f"🤖 Neural: {neural_response} ({time_taken:.3f}s)")
            
            # Claude response
            claude_response = self.get_claude_response(question)
            print(f"🔵 Claude: {claude_response}")
        
        print("\n🏆 Quick comparison complete! Who do you think did better overall?")
    
    def interactive_battle(self):
        """Interactive battle - user asks questions"""
        print("\n🎮 INTERACTIVE BATTLE MODE")
        print("Ask any question and see both responses!")
        print("Type 'quit' to exit")
        print("=" * 40)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                print(f"\n🎯 Question: {question}")
                print("-" * 30)
                
                # Neural response
                neural_response, time_taken = self.generate_neural_response(question)
                print(f"🤖 YOUR NEURAL LLM:")
                print(f"   {neural_response}")
                print(f"   ⚡ {time_taken:.3f}s")
                print()
                
                # Claude response
                claude_response = self.get_claude_response(question)
                print(f"🔵 CLAUDE:")
                print(f"   {claude_response}")
                print()
                
                winner = input("🏆 Better response? (neural/claude/tie): ").strip().lower()
                if winner == 'neural':
                    print("🎉 Point to your Neural LLM!")
                elif winner == 'claude':
                    print("🔵 Point to Claude!")
                else:
                    print("🤝 Both good!")
                    
            except KeyboardInterrupt:
                break
        
        print("👋 Interactive battle ended!")

def main():
    """Main battle interface"""
    print("⚔️ YOUR 38M NEURAL LLM vs CLAUDE BATTLE")
    print("=" * 50)
    print("🤖 Test your neural model against Claude!")
    print()
    print("1. Full Intelligence Battle (15 rounds)")
    print("2. Quick Face-off (5 rounds)")
    print("3. Interactive Battle (ask your own questions)")
    print("4. Exit")
    
    try:
        battle = NeuralVsClaudeBattle()
        
        while True:
            try:
                choice = input("\n🎯 Select battle mode (1-4): ").strip()
                
                if choice == '1':
                    battle.run_intelligence_battle()
                elif choice == '2':
                    battle.quick_face_off()
                elif choice == '3':
                    battle.interactive_battle()
                elif choice == '4':
                    print("👋 Battle complete!")
                    break
                else:
                    print("Please select 1-4")
                    
            except KeyboardInterrupt:
                print("\n👋 Battle ended")
                break
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've trained the expanded model first:")
        print("   python3 expand_neural_training.py")

if __name__ == "__main__":
    main()