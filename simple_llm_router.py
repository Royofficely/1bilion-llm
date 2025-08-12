#!/usr/bin/env python3
"""
SIMPLE LLM ROUTER - Working neural routing system
Learns to route queries to specialized agents
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import random

class SimpleRouterLLM(nn.Module):
    """Simple neural network for routing decisions"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 5)  # 5 agents
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use last hidden state
        output = self.classifier(self.dropout(hidden[-1]))  # (batch, num_agents)
        return output

class SimpleTokenizer:
    """Simple word-based tokenizer for routing"""
    
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        words = set()
        for text in texts:
            words.update(text.lower().split())
        
        # Add special tokens
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        
        # Add words
        for i, word in enumerate(sorted(words), 2):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
        
        self.vocab_size = len(self.word_to_id)
        print(f"📚 Built vocabulary: {self.vocab_size} words")
        
    def encode(self, text, max_length=10):
        """Encode text to token IDs"""
        words = text.lower().split()[:max_length]
        ids = [self.word_to_id.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad to max_length
        while len(ids) < max_length:
            ids.append(0)  # 0 is <PAD>
            
        return ids

class LLMRouter:
    """Neural LLM Router that learns to route queries"""
    
    def __init__(self):
        self.agents = {
            "math_agent": 0,
            "python_agent": 1, 
            "text_agent": 2,
            "knowledge_agent": 3,
            "web_agent": 4
        }
        
        self.agent_names = list(self.agents.keys())
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
    def create_training_data(self):
        """Create training data for routing"""
        data = [
            # Math queries
            ("what is 17 times 23", "math_agent"),
            ("calculate 2 plus 2", "math_agent"),
            ("solve 15 divided by 3", "math_agent"),
            ("fibonacci sequence", "math_agent"),
            ("factorial of 5", "math_agent"),
            ("pattern 2 4 6 8", "math_agent"),
            
            # Python queries
            ("write python function", "python_agent"),
            ("python code to sort", "python_agent"),
            ("create python script", "python_agent"),
            ("python algorithm", "python_agent"),
            ("def function python", "python_agent"),
            
            # Text queries
            ("reverse the word hello", "text_agent"),
            ("first letter of apple", "text_agent"),
            ("reverse palindrome", "text_agent"),
            ("count letters in word", "text_agent"),
            ("uppercase this text", "text_agent"),
            
            # Knowledge queries  
            ("what is gravity", "knowledge_agent"),
            ("explain photosynthesis", "knowledge_agent"),
            ("capital of france", "knowledge_agent"),
            ("why is sky blue", "knowledge_agent"),
            ("define machine learning", "knowledge_agent"),
            
            # Web queries
            ("bitcoin price today", "web_agent"),
            ("news about israel", "web_agent"),
            ("weather in bangkok", "web_agent"),
            ("current events", "web_agent"),
            ("latest news", "web_agent")
        ]
        
        return data
    
    def train_router(self, epochs=100):
        """Train the routing model"""
        print("🚀 TRAINING LLM ROUTER")
        print("=" * 40)
        
        # Create training data
        training_data = self.create_training_data()
        
        # Build tokenizer
        queries = [item[0] for item in training_data]
        self.tokenizer.build_vocab(queries)
        
        # Create model
        self.model = SimpleRouterLLM(self.tokenizer.vocab_size)
        self.model.to(self.device)
        
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Prepare training data
        X = torch.tensor([self.tokenizer.encode(query) for query, _ in training_data])
        y = torch.tensor([self.agents[agent] for _, agent in training_data])
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"🎯 Training on {len(training_data)} examples for {epochs} epochs")
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                accuracy = self.calculate_accuracy(X, y)
                print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.1f}%")
        
        # Final accuracy
        final_accuracy = self.calculate_accuracy(X, y)
        print(f"✅ Training complete! Final accuracy: {final_accuracy:.1f}%")
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'agents': self.agents
        }, 'simple_router_model.pt')
        print("💾 Model saved!")
    
    def calculate_accuracy(self, X, y):
        """Calculate prediction accuracy"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean() * 100
        self.model.train()
        return accuracy.item()
    
    def route_query(self, query):
        """Route a query to the best agent"""
        if self.model is None:
            return "knowledge_agent", f"Answer: {query}"
        
        # Encode query
        encoded = torch.tensor([self.tokenizer.encode(query)]).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(encoded)
            _, predicted = torch.max(outputs, 1)
            agent_idx = predicted.item()
        
        agent_name = self.agent_names[agent_idx]
        
        # Generate prompt based on agent
        prompts = {
            "math_agent": f"Calculate step by step: {query}",
            "python_agent": f"Execute Python code: {query}",
            "text_agent": f"Process text: {query}",
            "knowledge_agent": f"Explain with facts: {query}",
            "web_agent": f"Search web for: {query}"
        }
        
        return agent_name, prompts[agent_name]

def test_router():
    """Test the router"""
    print("🧪 TESTING SIMPLE LLM ROUTER")
    print("=" * 50)
    
    # Create and train router
    router = LLMRouter()
    router.train_router(epochs=200)
    
    # Test queries
    test_queries = [
        "What is 17 times 23?",
        "Write Python function to sort list",
        "Reverse the word hello", 
        "What is the capital of France?",
        "Bitcoin price today",
        "Calculate factorial of 7",
        "Python code for fibonacci",
        "Why is the sky blue?"
    ]
    
    print("\n📋 ROUTING TEST RESULTS:")
    print("-" * 50)
    
    for query in test_queries:
        agent, prompt = router.route_query(query)
        print(f"Query: {query}")
        print(f"→ Agent: {agent}")
        print(f"→ Prompt: {prompt}")
        print()
    
    return router

if __name__ == "__main__":
    router = test_router()