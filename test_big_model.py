#!/usr/bin/env python3
"""
Test the 38M parameter expanded neural model
"""

import torch
from real_neural_llm import RealNeuralLLM, SimpleTokenizer

def test_big_model():
    print("🔥 LOADING 38M PARAMETER NEURAL LLM")
    
    # Load the expanded model
    checkpoint = torch.load('expanded_neural_llm_checkpoint.pt')
    config = checkpoint['config']
    tokenizer_vocab = checkpoint['tokenizer_vocab']
    
    print(f"📊 Model: {checkpoint['total_parameters']:,} parameters")
    print(f"🧠 Config: {config}")
    
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
    print("✅ 38M parameter model loaded!")
    
    # Test various prompts
    test_prompts = [
        "hello world",
        "artificial intelligence", 
        "machine learning",
        "python programming",
        "neural networks",
        "deep learning",
        "how are you",
        "good morning",
        "thank you"
    ]
    
    print("\n🎯 TESTING 38M PARAMETER GENERATION:")
    print("=" * 60)
    
    with torch.no_grad():
        for prompt in test_prompts:
            print(f"\n🔥 Prompt: '{prompt}'")
            
            try:
                # Encode prompt
                input_tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([input_tokens], device=device)
                
                # Generate with different temperatures
                for temp in [0.7, 1.0]:
                    generated = model.generate(
                        input_ids,
                        max_length=30,
                        temperature=temp,
                        do_sample=True
                    )
                    
                    generated_text = tokenizer.decode(generated[0].tolist())
                    print(f"   T={temp}: {generated_text}")
                    
            except Exception as e:
                print(f"   Error: {e}")
    
    print("\n🏆 38M PARAMETER MODEL TESTING COMPLETE!")
    
    # Interactive mode
    print("\n🚀 Interactive mode (type 'quit' to exit):")
    while True:
        try:
            prompt = input("\n💭 Enter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if not prompt:
                continue
                
            input_tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([input_tokens], device=device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_length=40,
                    temperature=0.8,
                    do_sample=True
                )
            
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"🤖 38M Model: {generated_text}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("👋 Thanks for testing your 38M parameter neural LLM!")

if __name__ == "__main__":
    test_big_model()