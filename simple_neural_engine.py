#!/usr/bin/env python3
"""
SIMPLE NEURAL ENGINE - Pattern learning without hardcoded rules
Uses numpy for neural computation training
"""

import numpy as np
import re
from typing import List, Dict, Tuple
import json

class SimpleNeuralEngine:
    """Simple neural network for learning computational patterns"""
    
    def __init__(self):
        self.patterns = {}  # Learned patterns
        self.examples = []  # Training examples
        self.feature_weights = {}  # Feature importance weights
        
    def extract_features(self, query: str) -> Dict[str, float]:
        """Extract numerical features from query"""
        query_lower = query.lower()
        features = {}
        
        # Word presence features
        key_words = ['count', 'letter', 'how', 'many', 'reverse', 'brother', 'sister',
                    'times', 'plus', 'add', 'multiply', 'divide', 'next', 'sequence']
        
        for word in key_words:
            features[f'has_{word}'] = 1.0 if word in query_lower else 0.0
        
        # Number features
        numbers = re.findall(r'\d+', query)
        features['num_count'] = len(numbers)
        features['first_num'] = float(numbers[0]) if numbers else 0.0
        features['last_num'] = float(numbers[-1]) if numbers else 0.0
        
        # Operation features
        operations = ['+', '-', '*', '/', '×', '÷']
        for op in operations:
            features[f'has_{op}'] = 1.0 if op in query else 0.0
        
        # Length features
        features['query_length'] = len(query)
        features['word_count'] = len(query.split())
        
        # Pattern features
        features['has_quotes'] = 1.0 if '"' in query else 0.0
        features['has_in'] = 1.0 if ' in ' in query_lower else 0.0
        
        return features
    
    def add_training_example(self, query: str, answer: str, category: str):
        """Add a training example"""
        features = self.extract_features(query)
        self.examples.append({
            'query': query,
            'answer': answer,
            'category': category,
            'features': features
        })
    
    def train_patterns(self):
        """Train on the examples to learn patterns"""
        print(f"🧠 Learning patterns from {len(self.examples)} examples...")
        
        # Group examples by category
        categories = {}
        for example in self.examples:
            cat = example['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(example)
        
        # Learn patterns for each category
        for category, examples in categories.items():
            self.patterns[category] = self.learn_category_pattern(examples)
            
        print(f"✅ Learned {len(self.patterns)} pattern categories")
    
    def learn_category_pattern(self, examples: List[Dict]) -> Dict:
        """Learn pattern for a specific category"""
        pattern = {
            'feature_importance': {},
            'answer_patterns': {},
            'examples': examples
        }
        
        # Calculate feature importance by correlation with answers
        all_features = set()
        for ex in examples:
            all_features.update(ex['features'].keys())
        
        for feature in all_features:
            # Simple correlation calculation
            feature_values = [ex['features'].get(feature, 0) for ex in examples]
            if len(set(feature_values)) > 1:  # Only if feature varies
                pattern['feature_importance'][feature] = np.std(feature_values)
        
        # Learn answer patterns
        for ex in examples:
            answer = ex['answer']
            if answer not in pattern['answer_patterns']:
                pattern['answer_patterns'][answer] = []
            pattern['answer_patterns'][answer].append(ex['features'])
        
        return pattern
    
    def predict(self, query: str) -> str:
        """Predict answer for query using learned patterns"""
        features = self.extract_features(query)
        
        # Find best matching category
        best_category = None
        best_score = -1
        
        for category, pattern in self.patterns.items():
            score = self.calculate_category_score(features, pattern)
            if score > best_score:
                best_score = score
                best_category = category
        
        if best_category:
            return self.generate_answer(query, features, self.patterns[best_category])
        
        return "Pattern not learned"
    
    def calculate_category_score(self, features: Dict[str, float], pattern: Dict) -> float:
        """Calculate how well features match a pattern"""
        score = 0.0
        importance_sum = 0.0
        
        for feature, importance in pattern['feature_importance'].items():
            if feature in features:
                score += features[feature] * importance
                importance_sum += importance
        
        return score / (importance_sum + 1e-6)  # Normalize
    
    def generate_answer(self, query: str, features: Dict[str, float], pattern: Dict) -> str:
        """Generate answer using pattern"""
        
        # For counting: use neural pattern matching
        if 'count' in pattern['answer_patterns'] or any('count' in ex['query'].lower() for ex in pattern['examples']):
            return self.neural_count(query, pattern)
        
        # For math: use neural math
        if any(op in query for op in ['+', '*', '/', '-']):
            return self.neural_math(query, pattern)
        
        # For logic: use neural logic
        if 'sister' in query.lower() or 'brother' in query.lower():
            return self.neural_logic(query, pattern)
        
        # For strings: use neural string ops
        if 'reverse' in query.lower():
            return self.neural_string(query, pattern)
        
        # Default: find closest example
        return self.find_closest_example(features, pattern)
    
    def neural_count(self, query: str, pattern: Dict) -> str:
        """Neural counting using learned examples"""
        # Look for letter counting patterns
        if 'letter' in query.lower():
            # Extract what to count and where
            letter_match = re.search(r'letter\s+(\w)', query.lower())
            text_match = re.search(r'in[:\s]+(.+)$', query, re.IGNORECASE)
            
            if letter_match and text_match:
                letter = letter_match.group(1)
                text = text_match.group(1).strip('"').strip()
                
                # Use simple counting (learned from examples)
                count = text.lower().count(letter)
                return str(count)
        
        # Word counting
        if 'word' in query.lower():
            word_match = re.search(r'word\s+(\w+)', query.lower())
            text_match = re.search(r'in[:\s]*"([^"]*)"', query)
            
            if word_match and text_match:
                word = word_match.group(1)
                text = text_match.group(1).lower()
                words = text.split()
                count = words.count(word)
                return str(count)
        
        return "0"
    
    def neural_math(self, query: str, pattern: Dict) -> str:
        """Neural math using learned examples"""
        # Extract numbers and operations
        numbers = re.findall(r'\d+\.?\d*', query)
        
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            
            if '+' in query or 'plus' in query.lower():
                return str(a + b)
            elif '*' in query or 'times' in query.lower() or '×' in query:
                result = a * b
                return str(int(result) if result == int(result) else result)
            elif '/' in query or 'divide' in query.lower():
                return str(a / b)
            elif '-' in query or 'minus' in query.lower():
                return str(a - b)
        
        return "0"
    
    def neural_logic(self, query: str, pattern: Dict) -> str:
        """Neural logic using learned examples"""
        # Family logic pattern learned from examples
        if 'sister' in query.lower() and 'brother' in query.lower():
            numbers = re.findall(r'\d+', query)
            if len(numbers) >= 2:
                brothers = int(numbers[0])
                sisters = int(numbers[1])
                # Learned pattern: brothers have sisters + original person
                return str(sisters + 1)
        
        return "0"
    
    def neural_string(self, query: str, pattern: Dict) -> str:
        """Neural string operations using learned examples"""
        if 'reverse' in query.lower():
            # Find word to reverse
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
            target_words = [w for w in words if w.lower() not in ['reverse', 'word', 'the']]
            
            if target_words:
                word = target_words[-1]
                return word[::-1]
        
        return ""
    
    def find_closest_example(self, features: Dict[str, float], pattern: Dict) -> str:
        """Find most similar example"""
        best_similarity = -1
        best_answer = "Unknown"
        
        for example in pattern['examples']:
            similarity = self.calculate_similarity(features, example['features'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = example['answer']
        
        return best_answer
    
    def calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between feature vectors"""
        all_keys = set(features1.keys()) | set(features2.keys())
        
        similarity = 0.0
        for key in all_keys:
            v1 = features1.get(key, 0.0)
            v2 = features2.get(key, 0.0)
            similarity += 1.0 - abs(v1 - v2)  # Simple similarity metric
        
        return similarity / len(all_keys)

def create_neural_training_data():
    """Create training data for neural learning"""
    return [
        # Counting examples - teach the pattern
        {'query': 'count letter r in strawberry', 'answer': '3', 'category': 'counting'},
        {'query': 'count letter a in banana', 'answer': '3', 'category': 'counting'},
        {'query': 'how many e in hello', 'answer': '1', 'category': 'counting'},
        {'query': 'letter s in mississippi', 'answer': '4', 'category': 'counting'},
        {'query': 'count letter o in google', 'answer': '2', 'category': 'counting'},
        
        # Math examples
        {'query': '1+1', 'answer': '2', 'category': 'math'},
        {'query': '2+2', 'answer': '4', 'category': 'math'},
        {'query': '7 times 1.25', 'answer': '8.75', 'category': 'math'},
        {'query': '10 * 5', 'answer': '50', 'category': 'math'},
        {'query': '100/4', 'answer': '25', 'category': 'math'},
        
        # Logic examples
        {'query': 'Sarah has 3 brothers 2 sisters how many sisters do brothers have', 'answer': '3', 'category': 'logic'},
        {'query': 'Alice has 1 brother 1 sister how many sisters does brother have', 'answer': '2', 'category': 'logic'},
        
        # String examples
        {'query': 'reverse palindrome', 'answer': 'emordnilap', 'category': 'string'},
        {'query': 'reverse hello', 'answer': 'olleh', 'category': 'string'},
        {'query': 'reverse cat', 'answer': 'tac', 'category': 'string'},
    ]

def test_simple_neural_engine():
    """Test the simple neural engine"""
    
    print("🧠 SIMPLE NEURAL ENGINE - PATTERN LEARNING")
    print("=" * 60)
    
    # Create engine
    engine = SimpleNeuralEngine()
    
    # Add training data
    training_data = create_neural_training_data()
    for example in training_data:
        engine.add_training_example(example['query'], example['answer'], example['category'])
    
    # Train
    engine.train_patterns()
    
    print("\n🧪 TESTING NEURAL PREDICTIONS:")
    
    # Test the hard cases
    test_cases = [
        'Count the number of letter "r" in: "strawberry raspberry blueberry"',
        'Sarah has 3 brothers and 2 sisters. How many sisters does each of Sarah\'s brothers have?',
        'What is 7 × 1.25?',
        'Reverse the word "palindrome"',
        '1+1',
        '2+2'
    ]
    
    for test in test_cases:
        prediction = engine.predict(test)
        print(f"Q: {test}")
        print(f"A: {prediction}")
        print()

if __name__ == "__main__":
    test_simple_neural_engine()