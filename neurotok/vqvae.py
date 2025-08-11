import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 4096, embedding_dim: int = 128, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 256, hidden_dims: list = None, latent_dim: int = 128):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 512]
        
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim
        
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU()
        ))
        
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dims: list = None, out_channels: int = 256):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256]
        
        modules = []
        in_channels = latent_dim
        
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim
        
        modules.append(nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ))
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class VQVAE(nn.Module):
    def __init__(self, num_embeddings: int = 4096, embedding_dim: int = 128, 
                 hidden_dims: list = None, commitment_cost: float = 0.25):
        super().__init__()
        self.encoder = Encoder(in_channels=256, hidden_dims=hidden_dims, latent_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(latent_dim=embedding_dim, hidden_dims=hidden_dims, out_channels=256)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.permute(0, 2, 1)
        quantized, _, _ = self.vq(z)
        return quantized.permute(0, 2, 1)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        z = z.permute(0, 2, 1)
        quantized, vq_loss, perplexity = self.vq(z)
        quantized = quantized.permute(0, 2, 1)
        x_recon = self.decoder(quantized)
        
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss
        
        return {
            'recon': x_recon,
            'quantized': quantized,
            'loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'perplexity': perplexity
        }
    
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.permute(0, 2, 1)
        flat_z = z.view(-1, self.vq.embedding_dim)
        
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                    + torch.sum(self.vq.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.vq.embeddings.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices.view(z.shape[0], -1)
    
    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.vq.embeddings(codes)
        quantized = quantized.permute(0, 2, 1)
        return self.decoder(quantized)