#!/usr/bin/env python3
"""
AUTOMATED NEURAL LLM GENERATION
Continuous automatic text generation with your 38M parameter model
"""

import torch
import time
import random
from real_neural_llm import RealNeuralLLM, SimpleTokenizer

class AutomatedNeuralGenerator:
    def __init__(self, checkpoint_path='expanded_neural_llm_checkpoint.pt'):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.load_model()
        
    def load_model(self):
        """Load the 38M parameter neural model"""
        print("🔥 LOADING 38M PARAMETER NEURAL LLM FOR AUTOMATION")
        
        checkpoint = torch.load(self.checkpoint_path)
        config = checkpoint['config']
        tokenizer_vocab = checkpoint['tokenizer_vocab']
        
        print(f"📊 Model: {checkpoint['total_parameters']:,} parameters")
        
        # Recreate model and tokenizer
        self.model = RealNeuralLLM(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.vocab = tokenizer_vocab
        self.tokenizer.reverse_vocab = {v: k for k, v in tokenizer_vocab.items()}
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"🖥️  Device: {self.device}")
        print("✅ Automated neural generation ready!")
        
    def generate_text(self, prompt, max_length=50, temperature=0.8):
        """Generate text from prompt"""
        try:
            input_tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([input_tokens], device=self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(generated[0].tolist())
            return generated_text
            
        except Exception as e:
            return f"Generation error: {e}"
    
    def creative_prompts_generator(self):
        """Generate diverse creative prompts"""
        prompts = [
            # Technology & AI
            "artificial intelligence",
            "machine learning is",
            "neural networks can",
            "deep learning helps",
            "python programming",
            "computer science",
            "data analysis",
            "algorithm design",
            
            # Conversational
            "hello world",
            "good morning",
            "how are you",
            "thank you for",
            "please help me",
            "can you explain",
            "i would like",
            "let me tell you",
            
            # Creative & Descriptive
            "once upon a time",
            "in the future",
            "beautiful flowers",
            "the quick brown",
            "creative writing",
            "storytelling is",
            "imagination helps",
            "art and science",
            
            # Questions & Exploration
            "what is the",
            "how does this",
            "why do we",
            "where can i",
            "when will the",
            "who created this",
            "which approach works",
            "whether or not",
            
            # Learning & Education
            "learning new skills",
            "education is important",
            "knowledge helps us",
            "understanding complex",
            "solving problems requires",
            "critical thinking",
            "research shows that",
            "studies indicate",
            
            # Innovation & Future
            "innovation drives",
            "technology advances",
            "the future holds",
            "breakthrough discoveries",
            "scientific progress",
            "human potential",
            "endless possibilities",
            "transformative ideas"
        ]
        
        return prompts
    
    def automated_generation_session(self, duration_minutes=10, generation_interval=3):
        """Run automated generation session"""
        print(f"🚀 STARTING {duration_minutes}-MINUTE AUTOMATED NEURAL GENERATION")
        print(f"⏱️  Generating every {generation_interval} seconds")
        print("=" * 70)
        
        prompts = self.creative_prompts_generator()
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        generation_count = 0
        
        while time.time() < end_time:
            generation_count += 1
            
            # Select random prompt
            prompt = random.choice(prompts)
            
            # Generate with random temperature for variety
            temperature = random.uniform(0.6, 1.2)
            max_length = random.randint(30, 60)
            
            print(f"\n🧠 Generation #{generation_count}")
            print(f"💭 Prompt: '{prompt}'")
            print(f"🌡️  Temperature: {temperature:.2f}, Length: {max_length}")
            
            # Generate text
            start_gen = time.time()
            generated_text = self.generate_text(prompt, max_length, temperature)
            gen_time = time.time() - start_gen
            
            print(f"🤖 Neural Output: {generated_text}")
            print(f"⚡ Generation time: {gen_time:.3f}s")
            print("-" * 50)
            
            # Wait for next generation
            time.sleep(generation_interval)
        
        print(f"\n🎉 AUTOMATED SESSION COMPLETE!")
        print(f"📊 Total generations: {generation_count}")
        print(f"⏱️  Total time: {duration_minutes} minutes")
        print(f"🧠 Your 38M parameter neural LLM created {generation_count} unique texts!")
    
    def continuous_story_generation(self):
        """Generate a continuous story by chaining outputs"""
        print("📚 CONTINUOUS NEURAL STORY GENERATION")
        print("Your 38M parameter model will create an evolving story!")
        print("=" * 60)
        
        # Start with a story prompt
        story_prompts = [
            "once upon a time",
            "in a distant future",
            "the scientist discovered",
            "artificial intelligence began",
            "deep in the forest",
            "the neural network learned"
        ]
        
        current_prompt = random.choice(story_prompts)
        full_story = []
        
        print(f"📖 Story starting with: '{current_prompt}'")
        print()
        
        for chapter in range(5):
            print(f"📑 Chapter {chapter + 1}")
            print(f"💭 Current prompt: '{current_prompt}'")
            
            # Generate next part
            generated = self.generate_text(current_prompt, max_length=40, temperature=0.9)
            full_story.append(generated)
            
            print(f"🤖 Generated: {generated}")
            print()
            
            # Use the last few words as next prompt
            words = generated.split()
            if len(words) > 3:
                next_words = words[-3:]  # Last 3 words
                current_prompt = " ".join(next_words).replace('<end>', '').replace('<pad>', '').strip()
            else:
                current_prompt = random.choice(story_prompts)
            
            time.sleep(2)  # Pause between chapters
        
        print("📚 COMPLETE NEURAL STORY:")
        print("=" * 60)
        for i, part in enumerate(full_story, 1):
            print(f"Chapter {i}: {part}")
        print("=" * 60)
        print("🎉 Story generated entirely by your 38M parameter neural LLM!")
    
    def interactive_automation_menu(self):
        """Interactive menu for automation options"""
        print("🤖 AUTOMATED NEURAL LLM GENERATION MENU")
        print("=" * 50)
        print("1. Automated Generation Session (10 minutes)")
        print("2. Quick Generation Burst (2 minutes)")
        print("3. Continuous Story Generation")
        print("4. Custom Duration Session")
        print("5. Temperature Testing (same prompt, different temps)")
        print("6. Exit")
        
        while True:
            try:
                choice = input("\n🎯 Select option (1-6): ").strip()
                
                if choice == '1':
                    self.automated_generation_session(duration_minutes=10)
                elif choice == '2':
                    self.automated_generation_session(duration_minutes=2, generation_interval=5)
                elif choice == '3':
                    self.continuous_story_generation()
                elif choice == '4':
                    minutes = int(input("⏱️  Enter duration in minutes: "))
                    interval = int(input("🔄 Enter generation interval in seconds: "))
                    self.automated_generation_session(duration_minutes=minutes, generation_interval=interval)
                elif choice == '5':
                    self.temperature_testing()
                elif choice == '6':
                    print("👋 Thanks for using automated neural generation!")
                    break
                else:
                    print("❌ Please select 1-6")
                    
            except KeyboardInterrupt:
                print("\n👋 Automation stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def temperature_testing(self):
        """Test same prompt with different temperatures"""
        prompt = input("💭 Enter prompt for temperature testing: ").strip()
        if not prompt:
            prompt = "artificial intelligence"
        
        print(f"\n🌡️ TEMPERATURE TESTING: '{prompt}'")
        print("=" * 60)
        
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        
        for temp in temperatures:
            generated = self.generate_text(prompt, max_length=35, temperature=temp)
            print(f"T={temp}: {generated}")
            time.sleep(1)
        
        print("\n🔥 Notice how temperature affects creativity vs coherence!")

def main():
    """Main automation interface"""
    try:
        generator = AutomatedNeuralGenerator()
        generator.interactive_automation_menu()
    except FileNotFoundError:
        print("❌ Expanded model not found. Please train it first:")
        print("   python3 expand_neural_training.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()