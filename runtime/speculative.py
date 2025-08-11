import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class SpeculativeDrafter(nn.Module):
    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 128, 
                 num_layers: int = 2, chunk_size: int = 10):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        self.acceptance_threshold = 0.7
    
    def forward(self, codes: torch.Tensor, hidden: Optional[Tuple] = None):
        embedded = self.embedding(codes)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.output(output)
        return logits, hidden
    
    def draft(self, context: List[int], num_chunks: int = 1) -> List[List[int]]:
        if len(context) < 5:
            return self._random_draft(num_chunks)
        
        context_tensor = torch.tensor(context[-50:]).unsqueeze(0)
        
        drafts = []
        hidden = None
        
        with torch.no_grad():
            logits, hidden = self.forward(context_tensor, hidden)
            
            for _ in range(num_chunks):
                chunk = []
                
                for _ in range(self.chunk_size):
                    probs = F.softmax(logits[0, -1], dim=0)
                    
                    top_k = 5
                    top_probs, top_indices = torch.topk(probs, top_k)
                    
                    sampled_idx = torch.multinomial(top_probs, 1).item()
                    next_code = top_indices[sampled_idx].item()
                    
                    chunk.append(next_code)
                    
                    next_input = torch.tensor([[next_code]])
                    logits, hidden = self.forward(next_input, hidden)
                
                drafts.append(chunk)
        
        return drafts
    
    def _random_draft(self, num_chunks: int) -> List[List[int]]:
        drafts = []
        for _ in range(num_chunks):
            chunk = np.random.randint(0, self.vocab_size, self.chunk_size).tolist()
            drafts.append(chunk)
        return drafts
    
    def verify_draft(self, draft: List[int], target: List[int]) -> Tuple[bool, int]:
        if len(draft) != len(target):
            matches = 0
            for i in range(min(len(draft), len(target))):
                if draft[i] == target[i]:
                    matches += 1
                else:
                    break
            return False, matches
        
        matches = sum(1 for d, t in zip(draft, target) if d == t)
        acceptance_rate = matches / len(draft)
        
        return acceptance_rate >= self.acceptance_threshold, matches


class SpeculativeEngine:
    def __init__(self, main_model: nn.Module, draft_model: Optional[SpeculativeDrafter] = None):
        self.main_model = main_model
        self.draft_model = draft_model or SpeculativeDrafter()
        
        self.stats = {
            'total_drafts': 0,
            'accepted_drafts': 0,
            'total_tokens': 0,
            'draft_tokens': 0
        }
    
    def generate(self, context: List[int], max_length: int = 100) -> List[int]:
        generated = []
        
        while len(generated) < max_length:
            draft_chunks = self.draft_model.draft(context + generated, num_chunks=1)
            
            if draft_chunks:
                draft = draft_chunks[0]
                self.stats['total_drafts'] += 1
                self.stats['draft_tokens'] += len(draft)
                
                verified_tokens = self._verify_with_main_model(
                    context + generated, draft
                )
                
                if verified_tokens:
                    generated.extend(verified_tokens)
                    self.stats['accepted_drafts'] += len(verified_tokens) / len(draft)
                else:
                    main_tokens = self._generate_with_main_model(
                        context + generated, self.draft_model.chunk_size
                    )
                    generated.extend(main_tokens)
                
                self.stats['total_tokens'] += len(verified_tokens or main_tokens)
            else:
                main_tokens = self._generate_with_main_model(
                    context + generated, self.draft_model.chunk_size
                )
                generated.extend(main_tokens)
                self.stats['total_tokens'] += len(main_tokens)
        
        return generated[:max_length]
    
    def _verify_with_main_model(self, context: List[int], draft: List[int]) -> Optional[List[int]]:
        context_tensor = torch.tensor(context[-100:]).unsqueeze(0)
        
        with torch.no_grad():
            if hasattr(self.main_model, 'forward'):
                logits = self.main_model(context_tensor)
                
                verified = []
                for i, draft_token in enumerate(draft):
                    if i < logits.shape[1]:
                        probs = F.softmax(logits[0, i], dim=0)
                        if probs[draft_token] > 0.1:
                            verified.append(draft_token)
                        else:
                            break
                
                if len(verified) >= len(draft) * 0.5:
                    return verified
        
        return None
    
    def _generate_with_main_model(self, context: List[int], num_tokens: int) -> List[int]:
        generated = []
        context_tensor = torch.tensor(context[-100:]).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(num_tokens):
                if hasattr(self.main_model, 'forward'):
                    logits = self.main_model(context_tensor)
                    probs = F.softmax(logits[0, -1], dim=0)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = np.random.randint(0, self.draft_model.vocab_size)
                
                generated.append(next_token)
                context_tensor = torch.cat([
                    context_tensor[:, 1:],
                    torch.tensor([[next_token]])
                ], dim=1)
        
        return generated
    
    def get_speedup(self) -> float:
        if self.stats['total_tokens'] == 0:
            return 1.0
        
        draft_ratio = self.stats['draft_tokens'] / max(self.stats['total_tokens'], 1)
        accept_ratio = self.stats['accepted_drafts'] / max(self.stats['total_drafts'], 1)
        
        return 1.0 + (draft_ratio * accept_ratio * 0.5)
    
    def train_drafter(self, data: List[List[int]], epochs: int = 10):
        optimizer = torch.optim.Adam(self.draft_model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for sequence in data:
                if len(sequence) < self.draft_model.chunk_size + 1:
                    continue
                
                for i in range(len(sequence) - self.draft_model.chunk_size):
                    context = torch.tensor(sequence[i:i+self.draft_model.chunk_size]).unsqueeze(0)
                    target = torch.tensor(sequence[i+1:i+self.draft_model.chunk_size+1]).unsqueeze(0)
                    
                    optimizer.zero_grad()
                    logits, _ = self.draft_model(context)
                    loss = F.cross_entropy(logits.view(-1, self.draft_model.vocab_size), 
                                          target.view(-1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / max(len(data), 1)
                print(f"Drafter training epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def save_drafter(self, path: str):
        torch.save({
            'model_state_dict': self.draft_model.state_dict(),
            'stats': self.stats,
            'config': {
                'vocab_size': self.draft_model.vocab_size,
                'hidden_dim': self.draft_model.hidden_dim,
                'chunk_size': self.draft_model.chunk_size
            }
        }, path)
    
    def load_drafter(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.draft_model.load_state_dict(checkpoint['model_state_dict'])
        self.stats = checkpoint.get('stats', self.stats)