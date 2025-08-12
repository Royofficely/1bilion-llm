#!/usr/bin/env python3
"""
NEURAL COMPUTATION ENGINE - Trainable system that learns patterns
No hardcoded conditions - pure neural learning approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import json
from typing import List, Tuple, Dict

class NeuralComputationEngine(nn.Module):
    """Neural network that learns to solve computational problems"""
    
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512):
        super().__init__()
        
        # Character/token embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Classification heads for different computation types
        self.computation_classifier = nn.Linear(hidden_dim * 2, 5)  # count, math, logic, string, sequence
        
        # Output generators for each type
        self.count_head = nn.Linear(hidden_dim * 2, 100)  # numbers 0-99
        self.math_head = nn.Linear(hidden_dim * 2, 1000)  # math results  
        self.logic_head = nn.Linear(hidden_dim * 2, 10)   # logical answers
        self.string_head = nn.Linear(hidden_dim * 2, vocab_size)  # string outputs
        
        # Build vocabulary
        self.build_vocabulary()
        
        # Training data
        self.training_examples = []
    
    def build_vocabulary(self):
        """Build character-level vocabulary"""
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!:;+-*/=()[]"\'<>'
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(chars)
    
    def encode_text(self, text: str, max_len=200) -> torch.Tensor:
        """Encode text to tensor"""
        indices = [self.char_to_idx.get(char, 0) for char in text[:max_len]]
        indices += [0] * (max_len - len(indices))  # Pad
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_output(self, indices: List[int]) -> str:
        """Decode tensor output to text"""
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])
    
    def forward(self, input_ids):
        """Forward pass"""
        # Embed input
        embedded = self.embedding(input_ids)
        
        # LSTM processing
        lstm_out, _ = self.lstm(embedded)
        
        # Use last hidden state
        final_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]
        
        # Classify computation type
        comp_logits = self.computation_classifier(final_hidden)
        
        # Generate outputs for each type
        count_logits = self.count_head(final_hidden)
        math_logits = self.math_head(final_hidden)
        logic_logits = self.logic_head(final_hidden)
        string_logits = self.string_head(final_hidden)
        
        return {
            'computation_type': comp_logits,
            'count_output': count_logits,
            'math_output': math_logits,
            'logic_output': logic_logits,
            'string_output': string_logits
        }
    
    def add_training_example(self, query: str, answer: str, computation_type: str):
        """Add training example"""
        type_map = {'count': 0, 'math': 1, 'logic': 2, 'string': 3, 'sequence': 4}
        
        self.training_examples.append({
            'query': query,
            'answer': answer,
            'type': type_map.get(computation_type, 0)
        })
    
    def train_on_examples(self, epochs=100):
        """Train the neural network on examples"""
        if not self.training_examples:
            return
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Training on {len(self.training_examples)} examples...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for example in self.training_examples:
                optimizer.zero_grad()
                
                # Encode input
                input_tensor = self.encode_text(example['query']).unsqueeze(0)
                
                # Forward pass
                outputs = self.forward(input_tensor)
                
                # Compute losses
                type_target = torch.tensor([example['type']], dtype=torch.long)
                type_loss = criterion(outputs['computation_type'], type_target)
                
                # Answer-specific loss based on type
                answer_loss = 0
                if example['type'] == 0:  # count
                    try:
                        answer_num = int(example['answer'])
                        if answer_num < 100:
                            count_target = torch.tensor([answer_num], dtype=torch.long)
                            answer_loss = criterion(outputs['count_output'], count_target)
                    except:
                        pass
                elif example['type'] == 1:  # math
                    try:
                        answer_num = int(float(example['answer']) * 10)  # Scale for more precision
                        if 0 <= answer_num < 1000:
                            math_target = torch.tensor([answer_num], dtype=torch.long)
                            answer_loss = criterion(outputs['math_output'], math_target)
                    except:
                        pass
                
                total_loss_example = type_loss + answer_loss
                total_loss_example.backward()
                optimizer.step()
                
                total_loss += total_loss_example.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(self.training_examples)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    def predict(self, query: str) -> str:
        """Make prediction on query"""
        try:
            input_tensor = self.encode_text(query).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.forward(input_tensor)
                
                # Get computation type
                comp_type = torch.argmax(outputs['computation_type'], dim=1).item()
                
                # Generate answer based on type
                if comp_type == 0:  # count
                    count_pred = torch.argmax(outputs['count_output'], dim=1).item()
                    return str(count_pred)
                elif comp_type == 1:  # math
                    math_pred = torch.argmax(outputs['math_output'], dim=1).item()
                    return str(math_pred / 10.0)  # Unscale
                elif comp_type == 2:  # logic
                    logic_pred = torch.argmax(outputs['logic_output'], dim=1).item()
                    return str(logic_pred)
                else:
                    return "Neural prediction not available"
        
        except Exception as e:
            return f"Neural error: {str(e)[:50]}"

def create_training_data() -> List[Dict]:
    """Create training data for the neural network"""
    return [
        # Counting examples
        {'query': 'count letter r in strawberry', 'answer': '3', 'type': 'count'},
        {'query': 'how many r in strawberry', 'answer': '3', 'type': 'count'},
        {'query': 'number of a in banana', 'answer': '3', 'type': 'count'},
        {'query': 'letter e in hello', 'answer': '1', 'type': 'count'},
        {'query': 'count s in mississippi', 'answer': '4', 'type': 'count'},
        
        # Math examples
        {'query': '1+1', 'answer': '2', 'type': 'math'},
        {'query': '2+2', 'answer': '4', 'type': 'math'},
        {'query': '3*4', 'answer': '12', 'type': 'math'},
        {'query': '10/2', 'answer': '5', 'type': 'math'},
        {'query': '7 times 1.25', 'answer': '8.75', 'type': 'math'},
        {'query': '100 divided by 4', 'answer': '25', 'type': 'math'},
        
        # Logic examples
        {'query': 'sarah has 3 brothers 2 sisters how many sisters do brothers have', 'answer': '3', 'type': 'logic'},
        {'query': 'alice has 1 brother 1 sister how many sisters does brother have', 'answer': '2', 'type': 'logic'},
        {'query': 'family with 2 boys 1 girl how many sisters does each boy have', 'answer': '1', 'type': 'logic'},
        
        # String examples  
        {'query': 'reverse palindrome', 'answer': 'emordnilap', 'type': 'string'},
        {'query': 'reverse hello', 'answer': 'olleh', 'type': 'string'},
        {'query': 'backwards cat', 'answer': 'tac', 'type': 'string'},
        
        # Sequence examples
        {'query': '1 1 2 3 5 8 13 what next', 'answer': '21', 'type': 'sequence'},
        {'query': '2 4 6 8 what comes next', 'answer': '10', 'type': 'sequence'},
    ]

def test_neural_engine():
    """Test the neural computation engine"""
    
    print("🧠 NEURAL COMPUTATION ENGINE - PURE LEARNING")
    print("=" * 60)
    
    # Create and train model
    model = NeuralComputationEngine()
    
    # Add training data
    training_data = create_training_data()
    for example in training_data:
        model.add_training_example(example['query'], example['answer'], example['type'])
    
    print("🔥 TRAINING THE NEURAL NETWORK...")
    model.train_on_examples(epochs=50)  # Quick training
    
    print("\n🧪 TESTING NEURAL PREDICTIONS:")
    
    # Test cases
    test_cases = [
        'count letter r in strawberry raspberry blueberry',
        'sarah has 3 brothers 2 sisters how many sisters do brothers have',
        '7 times 1.25',
        'reverse palindrome',
        '1 1 2 3 5 8 13 what next'
    ]
    
    for test in test_cases:
        prediction = model.predict(test)
        print(f"Q: {test}")
        print(f"A: {prediction}")
        print()

if __name__ == "__main__":
    test_neural_engine()