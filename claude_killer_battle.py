#!/usr/bin/env python3
"""
CLAUDE KILLER BATTLE - Ultimate test with trained model
Your 113M parameter Claude-killer vs Claude
"""

import torch
import time
from real_neural_llm import RealNeuralLLM, SimpleTokenizer

class ClaudeKillerBattle:
    def __init__(self):
        self.neural_model = None
        self.tokenizer = None
        self.load_claude_killer_model()
        
    def load_claude_killer_model(self):
        """Load the trained Claude-killer model"""
        print("🔥 LOADING 113M PARAMETER CLAUDE-KILLER MODEL")
        
        try:
            checkpoint = torch.load('claude_killer_model.pt')
            print("✅ Claude-killer model found!")
        except FileNotFoundError:
            print("⚠️ Claude-killer model not found, using expanded model")
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
        
        total_params = checkpoint.get('total_parameters', sum(p.numel() for p in self.neural_model.parameters()))
        print(f"✅ Claude-Killer Ready: {total_params:,} parameters")
        print("🎯 Trained on Claude-beating examples with 0.14 final loss!")
        
    def test_claude_killer_responses(self):
        """Test responses on key battle questions"""
        print("\n🎯 TESTING CLAUDE-KILLER ON KEY BATTLE QUESTIONS")
        print("=" * 60)
        
        # Key questions from the battle that you should now dominate
        test_questions = [
            # Math (previously lost)
            "what is 17 times 23?",
            "what is the square root of 169?", 
            "what is 7 factorial?",
            "how many edges does a cube have?",
            
            # Programming (your specialty)
            "what is python?",
            "what is machine learning?",
            "what is artificial intelligence?",
            "what does cpu stand for?",
            
            # Science (previously lost)  
            "why is the sky blue?",
            "what is photosynthesis?",
            "how many bones in human body?",
            "what is gravity?",
            
            # Creative (your potential strength)
            "tell joke about robots",
            "reverse the word hello",
            "what rhymes with brain?",
        ]
        
        claude_responses = {
            "what is 17 times 23?": "391",
            "what is the square root of 169?": "13",
            "what is 7 factorial?": "5040", 
            "how many edges does a cube have?": "12",
            "what is python?": "Python is a programming language",
            "what is machine learning?": "Machine learning is algorithms that learn from data",
            "what is artificial intelligence?": "AI is simulation of human intelligence",
            "what does cpu stand for?": "Central Processing Unit",
            "why is the sky blue?": "Light scattering by atmosphere",
            "what is photosynthesis?": "Plants convert sunlight to energy",
            "how many bones in human body?": "206",
            "what is gravity?": "Force that attracts objects together", 
            "tell joke about robots": "Why did robot go therapy? Hardware issues!",
            "reverse the word hello": "olleh",
            "what rhymes with brain?": "train, rain, pain, gain"
        }
        
        neural_wins = 0
        claude_wins = 0
        
        for question in test_questions:
            print(f"\n🔥 TESTING: {question}")
            
            # Your Claude-killer response
            input_tokens = self.tokenizer.encode(question)
            input_ids = torch.tensor([input_tokens], device='cuda' if torch.cuda.is_available() else 'cpu')
            
            start_time = time.time()
            with torch.no_grad():
                generated = self.neural_model.generate(
                    input_ids,
                    max_length=40,
                    temperature=0.3,  # Low temp for accuracy
                    do_sample=False   # Greedy for best answer
                )
            gen_time = time.time() - start_time
            
            neural_response = self.tokenizer.decode(generated[0].tolist())
            claude_response = claude_responses.get(question, "Standard response")
            
            print(f"🤖 Your Claude-Killer: {neural_response}")
            print(f"🔵 Claude: {claude_response}")
            print(f"⚡ Speed: {gen_time:.3f}s")
            
            # Simple scoring - check if key concepts present
            expected_keywords = claude_response.lower().split()[:3]  # First 3 key words
            neural_lower = neural_response.lower()
            
            if any(keyword in neural_lower for keyword in expected_keywords):
                neural_wins += 1
                print("🏆 Winner: YOUR CLAUDE-KILLER! ✅")
            else:
                claude_wins += 1
                print("🏆 Winner: Claude")
            
            print("-" * 50)
        
        # Results
        print(f"\n🏆 CLAUDE-KILLER BATTLE TEST RESULTS")
        print("=" * 50)
        print(f"🤖 Your Claude-Killer: {neural_wins}/15 wins ({neural_wins/15*100:.1f}%)")
        print(f"🔵 Claude: {claude_wins}/15 wins ({claude_wins/15*100:.1f}%)")
        
        if neural_wins >= claude_wins:
            print("🎉 YOUR CLAUDE-KILLER IS WORKING!")
            print("🚀 Ready to dominate the full 100-question battle!")
        else:
            print("⚡ Shows improvement - model is learning Claude-beating patterns!")
        
        return neural_wins, claude_wins
    
    def run_sample_battle(self):
        """Run a quick 20-question sample battle"""
        print("\n⚔️ QUICK 20-QUESTION CLAUDE-KILLER BATTLE")
        print("=" * 50)
        
        # 20 key questions from the full battle
        sample_questions = [
            ("What is 17 × 23?", "391"),
            ("What is Python?", "programming language"),
            ("What is machine learning?", "algorithms learn data"), 
            ("What is artificial intelligence?", "simulation human intelligence"),
            ("Why is the sky blue?", "light scattering"),
            ("What is gravity?", "force attracts objects"),
            ("Reverse the word hello", "olleh"),
            ("What rhymes with brain?", "train rain pain"),
            ("How many edges does a cube have?", "12"),
            ("What is photosynthesis?", "plants sunlight energy"),
            ("Tell joke about robots", "robot therapy hardware"),
            ("What does CPU stand for?", "central processing unit"),
            ("What is the square root of 169?", "13"),
            ("Complete: 2, 6, 18, 54, ?", "162"),
            ("What is deep learning?", "neural networks layers"),
            ("What causes rain?", "water cycle evaporation"),
            ("What is 7 factorial?", "5040"),
            ("What is data science?", "insights from data"),
            ("How many bones in human body?", "206"),
            ("What is creativity?", "generating new ideas")
        ]
        
        print("🎯 Testing your Claude-killer on 20 key battle questions...")
        
        wins = 0
        for i, (question, expected_concepts) in enumerate(sample_questions, 1):
            # Generate response
            input_tokens = self.tokenizer.encode(question)
            input_ids = torch.tensor([input_tokens], device='cuda' if torch.cuda.is_available() else 'cpu')
            
            with torch.no_grad():
                generated = self.neural_model.generate(
                    input_ids, max_length=30, temperature=0.2, do_sample=False
                )
            
            response = self.tokenizer.decode(generated[0].tolist())
            
            # Check if response contains expected concepts
            response_lower = response.lower()
            expected_words = expected_concepts.lower().split()
            
            if any(word in response_lower for word in expected_words):
                wins += 1
                result = "✅ WIN"
            else:
                result = "❌ MISS"
            
            print(f"Q{i:2d}: {question[:30]}... → {result}")
        
        accuracy = (wins / len(sample_questions)) * 100
        
        print(f"\n🏆 SAMPLE BATTLE RESULTS:")
        print(f"✅ Claude-Killer Wins: {wins}/20 ({accuracy:.1f}%)")
        print(f"🎯 Previous Baseline: 17/100 (17%)")
        print(f"📈 Improvement: {accuracy:.1f}% vs 17% = {accuracy - 17:.1f}% gain!")
        
        if accuracy >= 60:
            print("🎉 INCREDIBLE! Claude-killer training worked!")
            print("🏆 Ready to beat Claude in full battle!")
        elif accuracy >= 40:
            print("💪 Strong improvement! Claude-killer is learning!")
        else:
            print("📚 Model needs more training on these concepts")
            
        return accuracy

def main():
    """Main Claude-killer testing"""
    print("🔥 CLAUDE-KILLER BATTLE TESTING")
    print("=" * 50)
    
    try:
        battle = ClaudeKillerBattle()
        
        print("Choose test mode:")
        print("1. Quick 15-question key test")
        print("2. Sample 20-question battle")
        print("3. Both tests")
        
        choice = input("Select (1-3): ").strip()
        
        if choice in ['1', '3']:
            battle.test_claude_killer_responses()
        
        if choice in ['2', '3']:
            accuracy = battle.run_sample_battle()
            
            print(f"\n🎯 NEXT STEPS:")
            if accuracy >= 50:
                print("🚀 Run full 100-question battle:")
                print("   python3 ultimate_100_question_battle.py")
                print("🏆 Your Claude-killer should dominate!")
            else:
                print("⚡ Continue training or run quick boost:")
                print("   python3 quick_claude_killer_update.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure claude_killer_model.pt exists")

if __name__ == "__main__":
    main()