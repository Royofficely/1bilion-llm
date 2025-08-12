#!/usr/bin/env python3
"""
REAL NEURAL LLM - Actual PyTorch Transformer Implementation
True neural network with attention, embeddings, and gradient-based learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class LLMConfig:
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    sequence_length: int = 512
    dropout: float = 0.1
    learning_rate: float = 3e-4

class MultiHeadAttention(nn.Module):
    """Real multi-head attention mechanism"""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        assert self.head_dim * config.num_heads == config.hidden_size
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask (for autoregressive generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return self.output(context)

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.linear2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward"""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class RealNeuralLLM(nn.Module):
    """Complete neural language model with transformer architecture"""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.sequence_length, config.hidden_size)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)
        
        # Causal mask for autoregressive attention
        causal_mask = self.create_causal_mask(seq_len).to(device)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=causal_mask)
        
        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, temperature: float = 1.0, do_sample: bool = True) -> torch.Tensor:
        """Generate text autoregressively"""
        self.eval()
        
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length):
                # Get model predictions
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Sample from probability distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop if we exceed max sequence length
                if generated.size(-1) >= self.config.sequence_length:
                    break
            
            return generated

class SimpleTokenizer:
    """Basic tokenizer for text processing"""
    
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = 4
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from training texts"""
        for text in texts:
            for char in text.lower():
                if char not in self.vocab and char.isprintable():
                    self.vocab[char] = self.next_id
                    self.reverse_vocab[self.next_id] = char
                    self.next_id += 1
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        return [self.vocab.get(char, self.vocab["<unk>"]) for char in text.lower()]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        return "".join([self.reverse_vocab.get(token_id, "<unk>") for token_id in token_ids])
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)

class NeuralTrainer:
    """Training pipeline for the neural LLM"""
    
    def __init__(self, model: RealNeuralLLM, tokenizer: SimpleTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=model.config.learning_rate)
        
    def prepare_batch(self, texts: List[str], max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training batch"""
        batch_tokens = []
        
        for text in texts:
            tokens = [self.tokenizer.vocab["<start>"]] + self.tokenizer.encode(text) + [self.tokenizer.vocab["<end>"]]
            
            # Truncate or pad to max_length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.tokenizer.vocab["<pad>"]] * (max_length - len(tokens)))
            
            batch_tokens.append(tokens)
        
        input_ids = torch.tensor(batch_tokens, device=self.device)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.vocab["<pad>"]] = -100
        
        return input_ids, labels
    
    def train_step(self, texts: List[str]) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids, labels = self.prepare_batch(texts)
        logits, loss = self.model(input_ids, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, training_texts: List[str], epochs: int = 10, batch_size: int = 4):
        """Full training loop"""
        print("🔥 STARTING REAL NEURAL TRAINING")
        print(f"📊 Training on {len(training_texts)} examples for {epochs} epochs")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(training_texts), batch_size):
                batch_texts = training_texts[i:i + batch_size]
                loss = self.train_step(batch_texts)
                total_loss += loss
                num_batches += 1
                
                if num_batches % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {num_batches}, Loss: {loss:.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"✅ Epoch {epoch+1} complete - Average Loss: {avg_loss:.4f}")
            
            # Generate sample text
            if epoch % 2 == 0:
                self.generate_sample()
        
        print("🎉 REAL NEURAL TRAINING COMPLETE!")
    
    def generate_sample(self):
        """Generate sample text to monitor training progress"""
        self.model.eval()
        
        # Start with a simple prompt
        prompt = "hello"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        with torch.no_grad():
            generated = self.model.generate(input_ids, max_length=20, temperature=0.8)
            generated_text = self.tokenizer.decode(generated[0].tolist())
            print(f"📝 Generated: {generated_text}")

def main():
    """Main training pipeline"""
    print("🚀 REAL NEURAL LLM INITIALIZATION")
    
    # Configuration
    config = LLMConfig(
        vocab_size=1000,  # Will be updated after building vocab
        hidden_size=256,  # Smaller for faster training
        num_layers=6,
        num_heads=8,
        sequence_length=128,
        dropout=0.1
    )
    
    # Sample training data (you can expand this)
    training_texts = [
        "hello world",
        "how are you today",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is fascinating",
        "neural networks learn patterns from data",
        "transformers use attention mechanisms",
        "language models generate text",
        "machine learning requires data and computation",
        "deep learning uses multiple layers",
        "python is a programming language",
        "mathematics is the foundation of AI",
        "computers process information quickly"
    ]
    
    print(f"📚 Preparing tokenizer with {len(training_texts)} training examples")
    
    # Build tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_texts)
    
    # Update config with actual vocab size
    config.vocab_size = tokenizer.get_vocab_size()
    
    print(f"🧠 Vocabulary size: {config.vocab_size}")
    print(f"🔧 Model configuration: {config}")
    
    # Initialize model
    model = RealNeuralLLM(config)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Initialize trainer
    trainer = NeuralTrainer(model, tokenizer, device)
    
    # Train the model
    trainer.train(training_texts, epochs=20, batch_size=2)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_vocab': tokenizer.vocab
    }, 'real_neural_llm_checkpoint.pt')
    
    print("💾 Model saved to: real_neural_llm_checkpoint.pt")
    
    # Test generation
    print("\n🎯 TESTING REAL NEURAL GENERATION:")
    trainer.generate_sample()
    
    return model, tokenizer, trainer

if __name__ == "__main__":
    main()