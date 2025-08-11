import torch
import torch.nn as nn
import json
from typing import List, Dict, Any, Optional
import re


class StructMini(nn.Module):
    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        
        self.schema_templates = {
            'product_v1': {
                'name': 'string',
                'price': 'number',
                'currency': 'string',
                'in_stock': 'boolean',
                'url': 'string'
            },
            'post_v1': {
                'title': 'string',
                'author': 'string',
                'date': 'string',
                'content': 'string',
                'tags': 'array'
            },
            'event_v1': {
                'name': 'string',
                'date': 'string',
                'time': 'string',
                'location': 'string',
                'description': 'string'
            }
        }
    
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(codes)
        output, _ = self.lstm(embedded)
        return self.output(output)
    
    def generate_json(self, schema_id: str, observations: Dict[str, Any], 
                     codes: Optional[List[int]] = None) -> str:
        schema = self.schema_templates.get(schema_id, {})
        
        if not schema:
            return json.dumps({'error': f'Unknown schema: {schema_id}'})
        
        result = {}
        
        for field, field_type in schema.items():
            value = self._extract_field(field, observations, field_type)
            
            if field_type == 'string':
                result[field] = str(value) if value is not None else ""
            elif field_type == 'number':
                result[field] = self._parse_number(value)
            elif field_type == 'boolean':
                result[field] = self._parse_boolean(value)
            elif field_type == 'array':
                result[field] = self._parse_array(value)
            else:
                result[field] = value
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _extract_field(self, field: str, observations: Dict[str, Any], field_type: str) -> Any:
        if field in observations:
            return observations[field]
        
        for key, value in observations.items():
            if field.lower() in key.lower() or key.lower() in field.lower():
                return value
        
        if isinstance(observations.get('content'), str):
            content = observations['content']
            
            if field == 'price':
                price_match = re.search(r'[$£€]\s*(\d+(?:\.\d+)?)', content)
                if price_match:
                    return price_match.group(1)
            
            elif field == 'date':
                date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', content)
                if date_match:
                    return date_match.group(0)
            
            elif field == 'time':
                time_match = re.search(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?', content, re.I)
                if time_match:
                    return time_match.group(0)
        
        if field_type == 'boolean':
            return False
        elif field_type == 'number':
            return 0
        elif field_type == 'array':
            return []
        else:
            return ""
    
    def _parse_number(self, value: Any) -> float:
        if value is None:
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            cleaned = re.sub(r'[^0-9.-]', '', value)
            try:
                return float(cleaned)
            except:
                return 0.0
        
        return 0.0
    
    def _parse_boolean(self, value: Any) -> bool:
        if value is None:
            return False
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.lower() in ['true', 'yes', '1', 'available', 'in stock']
        
        return bool(value)
    
    def _parse_array(self, value: Any) -> List[str]:
        if value is None:
            return []
        
        if isinstance(value, list):
            return [str(v) for v in value]
        
        if isinstance(value, str):
            if ',' in value:
                return [v.strip() for v in value.split(',')]
            elif ';' in value:
                return [v.strip() for v in value.split(';')]
            else:
                return [value]
        
        return []
    
    def validate_output(self, json_str: str, schema_id: str) -> bool:
        try:
            data = json.loads(json_str)
            schema = self.schema_templates.get(schema_id, {})
            
            for field in schema:
                if field not in data:
                    return False
            
            for field in data:
                if field not in schema:
                    return False
            
            return True
        except:
            return False
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim
        }, path)