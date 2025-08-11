import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
from pathlib import Path
import numpy as np
from typing import Optional
from vqvae import VQVAE


class TextDataset(Dataset):
    def __init__(self, data_path: str, chunk_size: int = 512):
        self.chunk_size = chunk_size
        with open(data_path, 'rb') as f:
            self.data = f.read()
        
        self.chunks = []
        for i in range(0, len(self.data) - chunk_size, chunk_size // 2):
            chunk = self.data[i:i+chunk_size]
            if len(chunk) == chunk_size:
                self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        tensor = torch.zeros(256, self.chunk_size)
        for i, byte in enumerate(chunk):
            tensor[byte, i] = 1.0
        return tensor


def train_epoch(model: VQVAE, dataloader: DataLoader, optimizer: optim.Optimizer, 
                device: torch.device, use_bf16: bool = False) -> dict:
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_bf16 else None
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        if use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(batch)
            scaler.scale(outputs['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch)
            outputs['loss'].backward()
            optimizer.step()
        
        total_loss += outputs['loss'].item()
        total_recon_loss += outputs['recon_loss'].item()
        total_vq_loss += outputs['vq_loss'].item()
        total_perplexity += outputs['perplexity'].item()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        'vq_loss': total_vq_loss / n,
        'perplexity': total_perplexity / n
    }


def evaluate_fidelity(model: VQVAE, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_fidelity = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            
            batch_binary = (batch > 0.5).float()
            recon_binary = (outputs['recon'] > 0.5).float()
            
            fidelity = (batch_binary == recon_binary).float().mean()
            total_fidelity += fidelity.item()
    
    return total_fidelity / len(dataloader)


def calculate_compression_ratio(model: VQVAE, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    original_size = 0
    compressed_size = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            codes = model.get_codes(batch)
            
            original_size += batch.shape[0] * batch.shape[2]
            compressed_size += codes.numel() * np.log2(model.vq.num_embeddings) / 8
    
    return original_size / compressed_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/text_small.txt')
    parser.add_argument('--codebook', type=int, default=4096)
    parser.add_argument('--hours', type=float, default=0.5)
    parser.add_argument('--save', type=str, default='checkpoints/neurotok.pt')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    if not os.path.exists(args.data):
        print(f"Creating sample data at {args.data}")
        os.makedirs(os.path.dirname(args.data) or '.', exist_ok=True)
        with open(args.data, 'w') as f:
            f.write("Sample text for VQ-VAE training. " * 1000)
    
    dataset = TextDataset(args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=True, num_workers=args.workers)
    
    model = VQVAE(num_embeddings=args.codebook).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    
    start_time = time.time()
    max_duration = args.hours * 3600
    epoch = 0
    
    print(f"Training for {args.hours} hours...")
    
    while time.time() - start_time < max_duration:
        epoch += 1
        metrics = train_epoch(model, dataloader, optimizer, device, args.bf16)
        
        if epoch % 10 == 0:
            fidelity = evaluate_fidelity(model, dataloader, device)
            compression = calculate_compression_ratio(model, dataloader, device)
            
            print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, "
                  f"Recon={metrics['recon_loss']:.4f}, VQ={metrics['vq_loss']:.4f}, "
                  f"Perplexity={metrics['perplexity']:.1f}, "
                  f"Fidelity={fidelity*100:.1f}%, Compression={compression:.2f}x")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'fidelity': fidelity,
                'compression': compression
            }, args.save)
    
    print(f"Training completed after {epoch} epochs")
    
    final_fidelity = evaluate_fidelity(model, dataloader, device)
    final_compression = calculate_compression_ratio(model, dataloader, device)
    print(f"Final metrics: Fidelity={final_fidelity*100:.1f}%, Compression={final_compression:.2f}x")


if __name__ == '__main__':
    main()