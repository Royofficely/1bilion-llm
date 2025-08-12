#!/usr/bin/env python3
"""
Test Real Neural LLM Generation
Load trained model and generate creative text
"""

import torch
from real_neural_llm import RealNeuralLLM, SimpleTokenizer, LLMConfig

def load_and_test_model():
    """Load saved model and test generation"""
    print("🔥 LOADING TRAINED NEURAL LLM")
    
    # Load checkpoint
    checkpoint = torch.load('real_neural_llm_checkpoint.pt')
    config = checkpoint['config']
    tokenizer_vocab = checkpoint['tokenizer_vocab']
    
    print(f"📊 Model Config: {config}")
    print(f"🧠 Vocabulary Size: {len(tokenizer_vocab)}")
    
    # Recreate model and tokenizer
    model = RealNeuralLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = tokenizer_vocab
    tokenizer.reverse_vocab = {v: k for k, v in tokenizer_vocab.items()}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"🖥️  Device: {device}")
    print("✅ Model loaded successfully!")
    
    # Test different prompts
    test_prompts = [
        "hello",
        "how are",
        "the quick",
        "artificial",
        "python is",
        "machine learning"
    ]
    
    print("\n🎯 TESTING NEURAL GENERATION:")
    print("=" * 50)
    
    with torch.no_grad():
        for prompt in test_prompts:
            print(f"\n🔥 Prompt: '{prompt}'")
            
            # Encode prompt
            input_tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([input_tokens], device=device)
            
            # Generate with different temperatures
            for temp in [0.5, 0.8, 1.2]:
                generated = model.generate(
                    input_ids, 
                    max_length=30, 
                    temperature=temp, 
                    do_sample=True
                )
                
                generated_text = tokenizer.decode(generated[0].tolist())
                print(f"   T={temp}: {generated_text}")
    
    return model, tokenizer

def interactive_generation():
    """Interactive text generation"""
    print("\n🚀 INTERACTIVE NEURAL GENERATION")
    print("Type prompts to see what your neural LLM generates!")
    print("(Type 'quit' to exit)")
    
    model, tokenizer = load_and_test_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    while True:
        prompt = input("\n💭 Enter prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
            
        if not prompt:
            continue
        
        try:
            # Encode and generate
            input_tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([input_tokens], device=device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_length=50,
                    temperature=0.8,
                    do_sample=True
                )
            
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"🤖 Generated: {generated_text}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Thanks for testing your neural LLM!")

if __name__ == "__main__":
    try:
        load_and_test_model()
        
        # Ask if user wants interactive mode
        print("\n🤔 Want to try interactive generation? (y/n)")
        choice = input().strip().lower()
        
        if choice in ['y', 'yes']:
            interactive_generation()
            
    except FileNotFoundError:
        print("❌ Model checkpoint not found. Please train the model first:")
        print("   python3 real_neural_llm.py")
    except Exception as e:
        print(f"❌ Error loading model: {e}")