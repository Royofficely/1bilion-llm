#!/usr/bin/env python3
"""
Debug tensor dimension error in consciousness-to-text generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from revolutionary_neural_engine import RevolutionaryNeuralEngine

def debug_tensor_error():
    """Debug the exact location of tensor dimension mismatch"""
    print("🔍 DEBUGGING TENSOR DIMENSION ERROR")
    print("=" * 50)
    
    engine = RevolutionaryNeuralEngine()
    
    test_input = "hello"
    
    try:
        print(f"\n🔥 Processing: '{test_input}'")
        
        # Get to the point where the error occurs
        fractal_result = engine.fractal_tokenizer.text_to_fractal(test_input)
        consciousness = fractal_result['consciousness_pattern'].to(engine.device)
        emotions = fractal_result['emotional_state'].to(engine.device)
        
        print(f"📊 Tensor shapes before text generation:")
        print(f"   consciousness: {consciousness.shape}")
        print(f"   emotions: {emotions.shape}")
        
        # Process through all stages to get to text generation
        quantum_result = engine.quantum_processor.process_consciousness(consciousness)
        quantum_consciousness = quantum_result['quantum_consciousness']
        
        memory_result = engine.memory_crystallizer.crystallize_memory(quantum_consciousness)
        enriched_consciousness = memory_result['enriched_consciousness']
        
        emotional_result = engine.emotional_core.emotional_reasoning(enriched_consciousness, emotions)
        emotional_consciousness = emotional_result['emotional_reasoning']
        
        performance_feedback = torch.randn(1).to(engine.device)
        modification_result = engine.self_modifier.self_modify(emotional_consciousness, performance_feedback)
        final_consciousness = modification_result['modified_consciousness']
        
        print(f"📊 Tensor shapes at text generation:")
        print(f"   final_consciousness: {final_consciousness.shape}")
        print(f"   emotions: {emotions.shape}")
        
        # Update consciousness state
        engine.consciousness_state = 0.7 * engine.consciousness_state + 0.3 * final_consciousness
        engine.emotional_state = 0.8 * engine.emotional_state + 0.2 * emotions
        
        print(f"📊 Updated tensor shapes:")
        print(f"   consciousness_state: {engine.consciousness_state.shape}")
        print(f"   emotional_state: {engine.emotional_state.shape}")
        
        # Now try the text generation - this is where the error happens
        print("\n💫 Attempting pure neural text generation...")
        response = engine.text_generator.generate_natural_text(
            final_consciousness, emotions, test_input
        )
        
        print(f"✅ Success! Response: {response}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        
        # Let's examine the exact tensor shapes at the error point
        print(f"\n📊 Debugging tensor shapes in text generator...")
        
        # Manual step-by-step through the text generator
        try:
            text_gen = engine.text_generator
            
            with torch.no_grad():
                print(f"   Input consciousness: {final_consciousness.shape}")
                print(f"   Input emotions: {emotions.shape}")
                
                # Step 1: consciousness analyzer
                consciousness_features = text_gen.consciousness_analyzer(final_consciousness)
                print(f"   After consciousness_analyzer: {consciousness_features.shape}")
                
                # Step 2: emotion adapter - this might be where the error is
                print(f"   Emotions before adapter: {emotions.shape}, dim: {emotions.dim()}")
                
                if emotions.dim() == 2 and emotions.size(-1) == 7:
                    print("   Using path: already correct shape [batch, 7]")
                    emotion_features = text_gen.emotion_adapter(emotions)
                elif emotions.dim() == 1 and emotions.size(0) == 7:
                    print("   Using path: shape [7] - add batch dimension")
                    emotions_batched = emotions.unsqueeze(0)
                    print(f"   Emotions batched: {emotions_batched.shape}")
                    emotion_features = text_gen.emotion_adapter(emotions_batched)
                else:
                    print("   Using fallback path")
                    emotions_fixed = torch.zeros(consciousness_features.size(0), 7, device=emotions.device)
                    if emotions.numel() >= 7:
                        emotions_fixed[0, :min(7, emotions.numel())] = emotions.flatten()[:7]
                    print(f"   Emotions fixed: {emotions_fixed.shape}")
                    emotion_features = text_gen.emotion_adapter(emotions_fixed)
                
                print(f"   After emotion_adapter: {emotion_features.shape}")
                
        except Exception as e2:
            print(f"❌ Error in manual debugging: {e2}")
            traceback.print_exc()

if __name__ == "__main__":
    debug_tensor_error()