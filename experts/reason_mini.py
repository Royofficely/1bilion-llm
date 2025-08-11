import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import json


class ReasonMini(nn.Module):
    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.plan_templates = [
            "1. Fetch data from {url}\n2. Extract {field}\n3. Format as JSON",
            "1. Query {endpoint}\n2. Parse response\n3. Validate schema\n4. Return result",
            "1. Get webpage {url}\n2. Select {selector}\n3. Extract text\n4. Structure data",
            "1. Call API {api}\n2. Process result\n3. Generate JSON",
            "1. Retrieve {resource}\n2. Transform data\n3. Output formatted",
            "1. Access {source}\n2. Filter results\n3. Create response"
        ]
    
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(codes)
        output, _ = self.lstm(embedded)
        return self.output(output)
    
    def generate_plan(self, codes: List[int], meta: Dict[str, Any]) -> str:
        if not codes or len(codes) < 10:
            return self._generate_simple_plan(meta)
        
        codes_tensor = torch.tensor(codes[:100]).unsqueeze(0)
        
        with torch.no_grad():
            if hasattr(self, 'lstm'):
                embedded = self.embedding(codes_tensor)
                output, _ = self.lstm(embedded)
                logits = self.output(output[0, -1])
                plan_idx = torch.argmax(logits).item() % len(self.plan_templates)
            else:
                plan_idx = sum(codes[:5]) % len(self.plan_templates)
        
        template = self.plan_templates[plan_idx]
        
        url = meta.get('url', 'https://api.example.com/data')
        selector = meta.get('selector', '.content')
        field = meta.get('field', 'data')
        
        plan = template.format(
            url=url,
            field=field,
            selector=selector,
            endpoint=url.split('/')[-1] if '/' in url else 'endpoint',
            api=url.split('/')[2] if url.startswith('http') else 'api',
            resource=field,
            source=url.split('/')[2] if url.startswith('http') else 'source'
        )
        
        if len(plan.split('\n')) > 6:
            plan = '\n'.join(plan.split('\n')[:6])
        
        return plan
    
    def _generate_simple_plan(self, meta: Dict[str, Any]) -> str:
        url = meta.get('url', '')
        wants_json = meta.get('wants_json', True)
        
        if url:
            if wants_json:
                return f"1. Fetch {url}\n2. Parse response\n3. Generate JSON"
            else:
                return f"1. Fetch {url}\n2. Extract content\n3. Return data"
        else:
            if wants_json:
                return "1. Process input\n2. Structure data\n3. Output JSON"
            else:
                return "1. Analyze query\n2. Generate response"
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }, path)