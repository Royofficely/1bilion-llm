import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from vqvae import VQVAE


class NeuroTokenizer:
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = VQVAE(num_embeddings=4096).to(self.device)
        self.chunk_size = 512
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load(checkpoint_path)
        
        self.model.eval()
    
    def encode(self, text: Union[str, bytes]) -> List[int]:
        if isinstance(text, str):
            text = text.encode('utf-8')
        
        if len(text) == 0:
            return []
        
        pad_len = (self.chunk_size - len(text) % self.chunk_size) % self.chunk_size
        text = text + b'\x00' * pad_len
        
        all_codes = []
        
        with torch.no_grad():
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i+self.chunk_size]
                tensor = torch.zeros(1, 256, self.chunk_size).to(self.device)
                
                for j, byte in enumerate(chunk):
                    tensor[0, byte, j] = 1.0
                
                codes = self.model.get_codes(tensor)
                all_codes.extend(codes[0].cpu().numpy().tolist())
        
        return all_codes
    
    def decode(self, codes: List[int]) -> str:
        if not codes:
            return ""
        
        codes_tensor = torch.tensor(codes).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model.from_codes(codes_tensor)
        
        reconstructed = reconstructed[0].cpu().numpy()
        
        bytes_list = []
        for i in range(reconstructed.shape[1]):
            byte_probs = reconstructed[:, i]
            byte_val = np.argmax(byte_probs)
            if byte_val > 0:
                bytes_list.append(byte_val)
        
        try:
            return bytes(bytes_list).decode('utf-8', errors='ignore').rstrip('\x00')
        except:
            return ""
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'chunk_size': self.chunk_size
        }, path)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'chunk_size' in checkpoint:
            self.chunk_size = checkpoint['chunk_size']
        print(f"Tokenizer loaded from {path}")
    
    def compress_ratio(self, text: str) -> float:
        original_size = len(text.encode('utf-8'))
        codes = self.encode(text)
        compressed_size = len(codes) * np.log2(4096) / 8
        return original_size / compressed_size if compressed_size > 0 else 0
    
    def fidelity(self, text: str) -> float:
        codes = self.encode(text)
        decoded = self.decode(codes)
        
        if not text:
            return 1.0
        
        matches = sum(1 for a, b in zip(text, decoded) if a == b)
        return matches / len(text)