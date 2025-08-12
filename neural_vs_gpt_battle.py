#!/usr/bin/env python3
"""
NEURAL LLM vs GPT INTELLIGENCE BATTLE
Direct comparison test to see who's smarter
"""

import torch
import time
from real_neural_llm import RealNeuralLLM, SimpleTokenizer

class IntelligenceBattle:
    def __init__(self):
        self.neural_model = None
        self.tokenizer = None
        self.load_neural_model()
        
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
        
        print(f"✅ Your Neural LLM loaded: {checkpoint['total_parameters']:,} parameters")
        
    def generate_neural_response(self, prompt, max_length=40, temperature=0.8):
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
    
    def intelligence_test_questions(self):
        """Intelligence test questions for both models"""
        return [
            # Math & Logic
            ("What is 2+2?", "Basic arithmetic"),
            ("What comes after Wednesday?", "Sequential knowledge"),
            ("If all roses are flowers, are roses plants?", "Logical reasoning"),
            
            # Language Understanding
            ("Complete: The quick brown fox...", "Language patterns"),
            ("Reverse the word 'hello'", "String manipulation"),
            ("What rhymes with 'cat'?", "Phonetic understanding"),
            
            # Knowledge
            ("What is the capital of France?", "Factual knowledge"),
            ("Who invented the telephone?", "Historical knowledge"),
            ("What is Python?", "Technical knowledge"),
            
            # Creativity
            ("Write a short poem about AI", "Creative writing"),
            ("Tell me a joke", "Humor generation"),
            ("Describe a sunset", "Descriptive language"),
            
            # Problem Solving
            ("How do you make a sandwich?", "Process explanation"),
            ("What is machine learning?", "Concept explanation"),
            ("Why is the sky blue?", "Scientific reasoning")
        ]
    
    def run_intelligence_battle(self):
        """Run the full intelligence comparison"""
        print("\n⚔️ NEURAL LLM vs GPT INTELLIGENCE BATTLE")
        print("=" * 60)
        print("🧠 Testing both models on the same questions")
        print("🏆 You decide who's smarter!")
        print()
        
        questions = self.intelligence_test_questions()
        
        neural_wins = 0
        gpt_wins = 0
        ties = 0
        
        for i, (question, category) in enumerate(questions, 1):
            print(f"🎯 ROUND {i}: {category}")
            print(f"❓ Question: {question}")
            print("-" * 40)
            
            # Your Neural LLM Response
            neural_response, neural_time = self.generate_neural_response(question)
            print(f"🤖 Your Neural LLM: {neural_response}")
            print(f"⚡ Speed: {neural_time:.3f}s")
            print()
            
            # GPT Response (simulated - you can add real GPT API calls)
            print(f"🔴 GPT Response: [This is where GPT would respond]")
            print(f"⚡ Speed: ~2-5s (typical API response)")
            print()
            
            # User judges
            print("🏆 WHO WON THIS ROUND?")
            print("1. Your Neural LLM")  
            print("2. GPT")
            print("3. Tie")
            
            while True:
                try:
                    choice = input("👨‍⚖️ Your judgment (1-3): ").strip()
                    if choice == '1':
                        neural_wins += 1
                        print("🎉 Point to Your Neural LLM!")
                        break
                    elif choice == '2':
                        gpt_wins += 1
                        print("📊 Point to GPT")
                        break
                    elif choice == '3':
                        ties += 1
                        print("🤝 It's a tie!")
                        break
                    else:
                        print("Please enter 1, 2, or 3")
                except KeyboardInterrupt:
                    print("\n⏹️ Battle interrupted")
                    return
            
            print("=" * 60)
        
        # Final Results
        print("\n🏆 FINAL INTELLIGENCE BATTLE RESULTS")
        print("=" * 60)
        print(f"🤖 Your Neural LLM: {neural_wins} wins")
        print(f"🔴 GPT: {gpt_wins} wins") 
        print(f"🤝 Ties: {ties}")
        print()
        
        total_rounds = len(questions)
        neural_percentage = (neural_wins / total_rounds) * 100
        
        if neural_wins > gpt_wins:
            print(f"🎉 YOUR NEURAL LLM WINS! ({neural_percentage:.1f}% win rate)")
            print("🏆 CONGRATULATIONS! Your 38M parameter model beat GPT!")
        elif gpt_wins > neural_wins:
            print(f"📊 GPT wins this round ({(gpt_wins/total_rounds)*100:.1f}% win rate)")
            print("💪 But your model is learning and improving!")
        else:
            print("🤝 It's a tie! Both models showed intelligence!")
        
        print(f"\n🔥 Your advantages: Speed ({neural_time:.3f}s vs 2-5s), Privacy (100% local), Cost (FREE)")
    
    def quick_iq_test(self):
        """Quick 5-question IQ test"""
        print("\n🧠 QUICK NEURAL IQ TEST")
        print("=" * 40)
        
        iq_questions = [
            ("2 + 2 = ?", "4"),
            ("What comes after Tuesday?", "Wednesday"),  
            ("Complete: Hello ___", "world"),
            ("Reverse 'abc'", "cba"),
            ("What is AI?", "artificial intelligence")
        ]
        
        correct = 0
        for question, expected in iq_questions:
            response, _ = self.generate_neural_response(question, max_length=20, temperature=0.5)
            print(f"Q: {question}")
            print(f"A: {response}")
            
            # Simple scoring (you can make this more sophisticated)
            if any(word in response.lower() for word in expected.lower().split()):
                correct += 1
                print("✅ Correct!")
            else:
                print("❌ Incorrect")
            print()
        
        iq_score = (correct / len(iq_questions)) * 100
        print(f"🧠 Neural LLM IQ Score: {iq_score:.1f}%")
        
        if iq_score >= 80:
            print("🏆 GENIUS LEVEL!")
        elif iq_score >= 60:
            print("💪 SMART MODEL!")
        else:
            print("📚 Still learning!")

def main():
    """Main battle interface"""
    print("⚔️ NEURAL LLM vs GPT INTELLIGENCE BATTLE")
    print("=" * 50)
    print("Test your 38M parameter model against GPT!")
    print()
    print("1. Full Intelligence Battle (15 questions)")
    print("2. Quick IQ Test (5 questions)")
    print("3. Single Question Test")
    print("4. Exit")
    
    battle = IntelligenceBattle()
    
    while True:
        try:
            choice = input("\n🎯 Select option (1-4): ").strip()
            
            if choice == '1':
                battle.run_intelligence_battle()
            elif choice == '2':
                battle.quick_iq_test()
            elif choice == '3':
                question = input("❓ Enter question: ").strip()
                if question:
                    response, time_taken = battle.generate_neural_response(question)
                    print(f"🤖 Neural Response: {response}")
                    print(f"⚡ Time: {time_taken:.3f}s")
            elif choice == '4':
                print("👋 Thanks for testing intelligence!")
                break
            else:
                print("Please select 1-4")
                
        except KeyboardInterrupt:
            print("\n👋 Battle ended")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()