#!/usr/bin/env python3
"""
Expand Neural Training with More Data and Bigger Model
Train a larger neural LLM for better text generation
"""

from real_neural_llm import RealNeuralLLM, SimpleTokenizer, LLMConfig, NeuralTrainer
import torch

def create_expanded_dataset():
    """Create larger, more diverse training dataset"""
    
    # Expanded training texts with more variety
    training_texts = [
        # Basic conversational
        "hello world",
        "how are you today",
        "good morning everyone",
        "have a nice day",
        "thank you very much",
        "you are welcome",
        "see you later",
        "goodbye for now",
        
        # Technology and AI
        "artificial intelligence is fascinating",
        "neural networks learn patterns from data",
        "transformers use attention mechanisms",
        "language models generate text",
        "machine learning requires data and computation",
        "deep learning uses multiple layers",
        "computers process information quickly",
        "algorithms solve complex problems",
        "data science extracts insights",
        "robotics combines AI with physical systems",
        
        # Programming
        "python is a programming language",
        "code should be clean and readable",
        "functions organize program logic",
        "variables store data values",
        "loops repeat code execution",
        "debugging finds and fixes errors",
        "testing ensures code quality",
        "version control tracks changes",
        
        # Science and knowledge
        "mathematics is the foundation of AI",
        "physics describes natural phenomena",
        "chemistry studies molecular interactions",
        "biology explores living systems",
        "astronomy observes celestial objects",
        "geology examines earth processes",
        "psychology understands human behavior",
        "economics analyzes resource allocation",
        
        # Creative and descriptive
        "the quick brown fox jumps over the lazy dog",
        "beautiful flowers bloom in spring gardens",
        "music fills the room with harmony",
        "books contain endless knowledge",
        "art expresses human creativity",
        "stories transport us to other worlds",
        "poetry captures emotions in words",
        "dance moves with rhythmic grace",
        
        # Questions and responses
        "what is your name",
        "where are you from",
        "how does this work",
        "why is this important",
        "when will this happen",
        "who can help with this",
        "can you explain this concept",
        "would you like to learn more",
        
        # Instructions and actions
        "please follow these steps",
        "first create the basic structure",
        "then add the details",
        "finally test the result",
        "remember to save your work",
        "always backup important data",
        "check for errors regularly",
        "optimize for better performance"
    ]
    
    return training_texts

def train_bigger_model():
    """Train larger neural model for better performance"""
    
    print("🚀 EXPANDED NEURAL LLM TRAINING")
    print("=" * 50)
    
    # Bigger model configuration
    config = LLMConfig(
        vocab_size=1000,  # Will be updated
        hidden_size=512,  # Increased from 256
        num_layers=12,    # Increased from 6
        num_heads=16,     # Increased from 8
        sequence_length=256,  # Increased from 128
        dropout=0.1,
        learning_rate=2e-4  # Slightly lower for stability
    )
    
    # Get expanded training data
    training_texts = create_expanded_dataset()
    
    print(f"📚 Training with {len(training_texts)} diverse examples")
    
    # Build tokenizer with larger vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_texts)
    
    # Update config with actual vocab size
    config.vocab_size = tokenizer.get_vocab_size()
    
    print(f"🧠 Vocabulary size: {config.vocab_size}")
    print(f"🔧 Model configuration: {config}")
    
    # Initialize bigger model
    model = RealNeuralLLM(config)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🔥 Trainable parameters: {trainable_params:,}")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Initialize trainer
    trainer = NeuralTrainer(model, tokenizer, device)
    
    # Train for more epochs with larger dataset
    print("\n🔥 STARTING EXPANDED TRAINING...")
    trainer.train(training_texts, epochs=50, batch_size=4)
    
    # Save bigger model
    checkpoint_path = 'expanded_neural_llm_checkpoint.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_vocab': tokenizer.vocab,
        'training_examples': len(training_texts),
        'total_parameters': total_params
    }, checkpoint_path)
    
    print(f"💾 Expanded model saved to: {checkpoint_path}")
    
    # Test generation with expanded model
    print("\n🎯 TESTING EXPANDED MODEL GENERATION:")
    test_prompts = [
        "artificial intelligence",
        "machine learning is",
        "python programming",
        "how are you",
        "the quick brown",
        "beautiful flowers"
    ]
    
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            input_tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([input_tokens], device=device)
            
            generated = model.generate(
                input_ids,
                max_length=40,
                temperature=0.7,
                do_sample=True
            )
            
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"🤖 '{prompt}' → {generated_text}")
    
    return model, tokenizer, trainer

if __name__ == "__main__":
    train_bigger_model()