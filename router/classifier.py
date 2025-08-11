import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np


class TaskClassifier(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.feature_extractor = SimpleFeatureExtractor()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)
    
    def classify(self, meta: Dict[str, Any]) -> Tuple[str, float]:
        features = self.feature_extractor.extract(meta)
        features_tensor = torch.tensor(features).unsqueeze(0).float()
        
        with torch.no_grad():
            probs = self.forward(features_tensor)
        
        class_idx = torch.argmax(probs).item()
        confidence = probs[0, class_idx].item()
        
        classes = ['simple', 'moderate', 'complex']
        return classes[class_idx], confidence


class SimpleFeatureExtractor:
    def __init__(self):
        self.feature_dim = 128
    
    def extract(self, meta: Dict[str, Any]) -> np.ndarray:
        features = np.zeros(self.feature_dim)
        
        query = meta.get('query', '')
        features[0] = len(query) / 500.0
        
        features[1] = 1.0 if meta.get('wants_json', False) else 0.0
        features[2] = 1.0 if meta.get('url', '') else 0.0
        features[3] = 1.0 if meta.get('selector', '') else 0.0
        
        keywords = {
            'fetch': 4, 'get': 5, 'extract': 6, 'parse': 7,
            'analyze': 8, 'compare': 9, 'transform': 10,
            'multiple': 11, 'all': 12, 'every': 13
        }
        
        for word, idx in keywords.items():
            if word in query.lower():
                features[idx] = 1.0
        
        if query:
            words = query.split()
            features[20] = len(words) / 100.0
            
            unique_words = len(set(words))
            features[21] = unique_words / max(len(words), 1)
        
        features[30] = np.random.random() * 0.1
        
        return features


class AmbiguityDetector:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.ambiguous_patterns = [
            'might', 'maybe', 'possibly', 'could be', 'or',
            'various', 'different', 'multiple options', 'depends'
        ]
    
    def is_ambiguous(self, query: str) -> bool:
        query_lower = query.lower()
        
        for pattern in self.ambiguous_patterns:
            if pattern in query_lower:
                return True
        
        if query.count('?') > 2:
            return True
        
        if len(query.split(' or ')) > 2:
            return True
        
        return False
    
    def get_ambiguity_score(self, query: str) -> float:
        score = 0.0
        query_lower = query.lower()
        
        for pattern in self.ambiguous_patterns:
            if pattern in query_lower:
                score += 0.2
        
        score += query.count('?') * 0.1
        score += (len(query.split(' or ')) - 1) * 0.15
        
        return min(1.0, score)