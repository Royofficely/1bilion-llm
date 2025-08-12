#!/usr/bin/env python3
"""
Pure Neural Only - Zero Demo Version
Shows only what your trained neural networks actually output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

class RealVQVAE(nn.Module):
    """Real VQ-VAE architecture matching training"""
    def __init__(self, num_embeddings=4096, embedding_dim=128):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU()
        )
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def encode_text(self, text: str):
        """Convert text to neural codes"""
        bytes_data = np.array([ord(c) for c in text[:256]], dtype=np.float32)
        if len(bytes_data) < 256:
            bytes_data = np.pad(bytes_data, (0, 256 - len(bytes_data)), 'constant', constant_values=0)
        
        input_tensor = torch.FloatTensor(bytes_data).unsqueeze(0).unsqueeze(0).repeat(1, 256, 1)
        
        if next(self.parameters()).is_cuda:
            input_tensor = input_tensor.cuda()
        
        with torch.no_grad():
            encoded = self.encoder(input_tensor)
            encoded = encoded.permute(0, 2, 1)
            flat_encoded = encoded.reshape(-1, self.embedding_dim)
            
            distances = (torch.sum(flat_encoded**2, dim=1, keepdim=True) + 
                        torch.sum(self.embeddings.weight**2, dim=1) - 
                        2 * torch.matmul(flat_encoded, self.embeddings.weight.t()))
            
            codes = torch.argmin(distances, dim=1)
            return codes.cpu().numpy().tolist()

class RealTransformer(nn.Module):
    """Real transformer matching training checkpoints"""
    def __init__(self, vocab_size=4096, hidden_dim=512, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def generate_tokens(self, input_codes, max_tokens=30):
        """Generate raw neural tokens"""
        with torch.no_grad():
            if len(input_codes) == 0:
                input_codes = [42]
            
            input_codes = input_codes[:50]  # Context limit
            input_tensor = torch.LongTensor([input_codes])
            
            if next(self.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            
            generated_tokens = []
            current_sequence = input_tensor
            
            for _ in range(max_tokens):
                embeddings = self.embedding(current_sequence)
                transformer_output = self.transformer(embeddings)
                logits = self.output_projection(transformer_output[0, -1, :])
                
                # Sample with low temperature for more deterministic output
                probs = F.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_tokens.append(next_token)
                
                new_token_tensor = torch.LongTensor([[next_token]])
                if current_sequence.is_cuda:
                    new_token_tensor = new_token_tensor.cuda()
                
                current_sequence = torch.cat([current_sequence, new_token_tensor], dim=1)
                
                if current_sequence.size(1) > 100:
                    current_sequence = current_sequence[:, -50:]
            
            return generated_tokens

class PureNeuralSystem:
    """Pure neural system - only real computation, zero demos"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        
        print("🧠 PURE NEURAL SYSTEM - ZERO DEMOS")
        print("=" * 45)
        print("Only shows real neural network outputs")
        print(f"Device: {self.device}")
        
        # Load models
        self.vqvae = RealVQVAE().to(self.device)
        self.transformer = RealTransformer(hidden_dim=512, num_layers=6).to(self.device)
        
        self.load_real_weights()
        
    def load_real_weights(self):
        """Load only real trained weights"""
        print("\n📂 Loading trained weights...")
        
        real_models_loaded = 0
        
        # VQ-VAE
        vqvae_path = self.checkpoint_dir / "neurotok.pt"
        if vqvae_path.exists():
            try:
                checkpoint = torch.load(vqvae_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    model_dict = self.vqvae.state_dict()
                    compatible = {k: v for k, v in checkpoint.items() 
                                if k in model_dict and model_dict[k].shape == v.shape}
                    
                    if compatible:
                        self.vqvae.load_state_dict(compatible, strict=False)
                        real_models_loaded += 1
                        print(f"✅ VQ-VAE: {len(compatible)} layers loaded from real checkpoint")
                    else:
                        print("❌ VQ-VAE: No compatible weights found")
                else:
                    print("❌ VQ-VAE: Checkpoint format incompatible")
            except Exception as e:
                print(f"❌ VQ-VAE: Load error - {str(e)[:50]}")
        else:
            print("❌ VQ-VAE: Checkpoint not found")
        
        # Transformer
        transformer_path = self.checkpoint_dir / "reason_mini.pt"
        if transformer_path.exists():
            try:
                checkpoint = torch.load(transformer_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    model_dict = self.transformer.state_dict()
                    compatible = {k: v for k, v in checkpoint.items() 
                                if k in model_dict and model_dict[k].shape == v.shape}
                    
                    if compatible:
                        self.transformer.load_state_dict(compatible, strict=False)
                        real_models_loaded += 1
                        print(f"✅ Transformer: {len(compatible)} layers loaded from real checkpoint")
                    else:
                        print("❌ Transformer: No compatible weights found")
                else:
                    print("❌ Transformer: Checkpoint format incompatible")
            except Exception as e:
                print(f"❌ Transformer: Load error - {str(e)[:50]}")
        else:
            print("❌ Transformer: Checkpoint not found")
        
        print(f"\n🎯 Status: {real_models_loaded}/2 models loaded with real weights")
        
        if real_models_loaded == 0:
            print("⚠️  WARNING: No real weights loaded - using random initialization")
            print("   Outputs will be random neural noise, not trained responses")
        elif real_models_loaded == 1:
            print("⚠️  PARTIAL: Only partial model loading succeeded")
            print("   Outputs will be mixed real/random neural computation")
        else:
            print("✅ SUCCESS: Real trained models loaded")
            print("   Outputs are from your $13.46 trained neural networks")
        
        # Set to eval mode
        self.vqvae.eval()
        self.transformer.eval()
    
    def process_input(self, text: str):
        """Process input through real neural networks only"""
        print(f"\n🔥 Input: '{text}'")
        
        start_time = time.time()
        
        # Step 1: Real VQ-VAE encoding
        print("⚡ Stage 1: VQ-VAE encoding...")
        neural_codes = self.vqvae.encode_text(text)
        vqvae_time = time.time() - start_time
        
        compression_ratio = len(text) / len(neural_codes) if neural_codes else 1.0
        print(f"   Output: {len(neural_codes)} neural codes")
        print(f"   Compression: {compression_ratio:.2f}x")
        print(f"   Time: {vqvae_time*1000:.1f}ms")
        
        # Step 2: Real transformer generation
        print("⚡ Stage 2: Transformer generation...")
        transformer_start = time.time()
        generated_tokens = self.transformer.generate_tokens(neural_codes, max_tokens=20)
        transformer_time = time.time() - transformer_start
        
        print(f"   Output: {len(generated_tokens)} tokens generated")
        print(f"   Time: {transformer_time*1000:.1f}ms")
        
        total_time = time.time() - start_time
        
        # Raw neural outputs only
        print("\n📊 RAW NEURAL OUTPUTS:")
        print(f"   VQ-VAE codes: {neural_codes[:10]}{'...' if len(neural_codes) > 10 else ''}")
        print(f"   Generated tokens: {generated_tokens}")
        
        # Basic statistics only
        if generated_tokens:
            token_stats = {
                'count': len(generated_tokens),
                'min': min(generated_tokens),
                'max': max(generated_tokens),
                'avg': sum(generated_tokens) / len(generated_tokens),
                'range': max(generated_tokens) - min(generated_tokens)
            }
            print(f"   Token statistics: {token_stats}")
        
        print(f"\n⏱️  Total neural computation: {total_time*1000:.1f}ms")
        
        return {
            'vqvae_codes': neural_codes,
            'generated_tokens': generated_tokens,
            'vqvae_time': vqvae_time,
            'transformer_time': transformer_time,
            'total_time': total_time,
            'compression_ratio': compression_ratio
        }

def main():
    """Pure neural interaction - no demos"""
    system = PureNeuralSystem()
    
    print("\n" + "="*50)
    print("🧠 PURE NEURAL INTERACTION")
    print("Shows only real neural network computation")
    print("No demos, no templates, no fake responses")
    print("Type 'quit' to exit")
    print("="*50)
    
    session_stats = {'inputs': 0, 'total_time': 0}
    
    while True:
        try:
            user_input = input(f"\n👤 Input: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                if session_stats['inputs'] > 0:
                    avg_time = session_stats['total_time'] / session_stats['inputs']
                    print(f"\n📊 Session: {session_stats['inputs']} inputs, avg {avg_time*1000:.0f}ms")
                print("🧠 Pure neural session ended")
                break
            
            # Process through real neural networks only
            result = system.process_input(user_input)
            
            session_stats['inputs'] += 1
            session_stats['total_time'] += result['total_time']
            
            print(f"\n✅ Neural processing complete")
            print(f"   Real computation: {result['total_time']*1000:.0f}ms")
            
        except KeyboardInterrupt:
            print(f"\n🧠 Neural session interrupted")
            break
        except Exception as e:
            print(f"❌ Neural error: {e}")
            continue

if __name__ == "__main__":
    main()