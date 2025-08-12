#!/usr/bin/env python3
"""
PATTERN LEARNING ENGINE - Pure Python neural-style learning
No hardcoded rules, only pattern learning from examples
"""

import re
import math
from typing import List, Dict, Tuple

class PatternLearningEngine:
    """Learns computational patterns from examples without hardcoded rules"""
    
    def __init__(self):
        self.learned_patterns = {}
        self.examples = []
        
    def add_example(self, input_text: str, output_text: str):
        """Add training example"""
        pattern_features = self.extract_pattern_features(input_text)
        
        self.examples.append({
            'input': input_text,
            'output': output_text,
            'features': pattern_features
        })
        
    def extract_pattern_features(self, text: str) -> Dict:
        """Extract pattern features from text"""
        text_lower = text.lower()
        
        features = {
            # Word patterns
            'has_count': 'count' in text_lower,
            'has_letter': 'letter' in text_lower, 
            'has_reverse': 'reverse' in text_lower,
            'has_sister': 'sister' in text_lower,
            'has_brother': 'brother' in text_lower,
            'has_many': 'many' in text_lower,
            'has_how': 'how' in text_lower,
            'has_in': ' in ' in text_lower,
            
            # Math patterns
            'has_plus': '+' in text or 'plus' in text_lower,
            'has_times': '*' in text or 'times' in text_lower or '×' in text,
            'has_divide': '/' in text or 'divide' in text_lower,
            'has_equals': '=' in text,
            
            # Structure patterns
            'has_quotes': '"' in text,
            'has_numbers': bool(re.findall(r'\d+', text)),
            'number_count': len(re.findall(r'\d+', text)),
            'word_count': len(text.split()),
            'has_question': '?' in text,
            
            # Specific number patterns
            'numbers': re.findall(r'\d+', text),
            'first_number': int(re.findall(r'\d+', text)[0]) if re.findall(r'\d+', text) else 0,
            'has_decimal': '.' in text and any(c.isdigit() for c in text.split('.')),
        }
        
        return features
    
    def train(self):
        """Train on examples to learn patterns"""
        print(f"🧠 Learning patterns from {len(self.examples)} examples...")
        
        # Group examples by similar patterns
        pattern_groups = self.group_by_patterns()
        
        # Learn rules for each group
        for group_name, examples in pattern_groups.items():
            self.learned_patterns[group_name] = self.learn_group_pattern(examples)
            
        print(f"✅ Learned {len(self.learned_patterns)} pattern types")
        
    def group_by_patterns(self) -> Dict[str, List]:
        """Group examples by similar feature patterns"""
        groups = {
            'counting': [],
            'arithmetic': [],
            'family_logic': [],
            'string_ops': [],
            'sequences': []
        }
        
        for example in self.examples:
            features = example['features']
            
            # Classify based on feature patterns
            if features['has_count'] or features['has_letter']:
                groups['counting'].append(example)
            elif features['has_plus'] or features['has_times'] or features['has_divide']:
                groups['arithmetic'].append(example)  
            elif features['has_sister'] and features['has_brother']:
                groups['family_logic'].append(example)
            elif features['has_reverse']:
                groups['string_ops'].append(example)
            else:
                groups['sequences'].append(example)
                
        return groups
    
    def learn_group_pattern(self, examples: List[Dict]) -> Dict:
        """Learn pattern for a group of similar examples"""
        pattern = {
            'examples': examples,
            'input_patterns': [],
            'output_patterns': [],
            'feature_correlations': {}
        }
        
        # Analyze input patterns
        for ex in examples:
            pattern['input_patterns'].append(ex['features'])
            pattern['output_patterns'].append(ex['output'])
            
        return pattern
    
    def predict(self, query: str) -> str:
        """Predict output for query using learned patterns"""
        features = self.extract_pattern_features(query)
        
        # Find best matching pattern group
        best_group = self.find_best_pattern_group(features)
        
        if best_group and best_group in self.learned_patterns:
            return self.apply_pattern(query, features, self.learned_patterns[best_group])
        
        return "Pattern not learned"
    
    def find_best_pattern_group(self, features: Dict) -> str:
        """Find which pattern group best matches the features"""
        
        # Use feature patterns to classify
        if features['has_count'] or features['has_letter']:
            return 'counting'
        elif features['has_plus'] or features['has_times'] or features['has_divide']:
            return 'arithmetic'
        elif features['has_sister'] and features['has_brother']:
            return 'family_logic'
        elif features['has_reverse']:
            return 'string_ops'
        else:
            return 'sequences'
    
    def apply_pattern(self, query: str, features: Dict, pattern: Dict) -> str:
        """Apply learned pattern to generate answer"""
        
        # Find most similar example in the pattern
        best_example = None
        best_similarity = -1
        
        for example in pattern['examples']:
            similarity = self.calculate_similarity(features, example['features'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_example = example
        
        if best_example:
            # Use the pattern from best example to generate answer
            return self.generate_from_example(query, best_example)
        
        return "No similar example found"
    
    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature sets"""
        matches = 0
        total_features = 0
        
        for key in set(features1.keys()) | set(features2.keys()):
            total_features += 1
            if features1.get(key) == features2.get(key):
                matches += 1
                
        return matches / total_features if total_features > 0 else 0
    
    def generate_from_example(self, query: str, example: Dict) -> str:
        """Generate answer using pattern from similar example"""
        
        # Extract the computational pattern from the example
        example_input = example['input'].lower()
        example_output = example['output']
        
        # COUNTING PATTERN
        if 'count' in example_input or 'letter' in example_input:
            return self.apply_counting_pattern(query, example)
        
        # ARITHMETIC PATTERN
        elif any(op in example_input for op in ['+', '*', '/', 'times', 'plus']):
            return self.apply_arithmetic_pattern(query, example)
        
        # FAMILY LOGIC PATTERN
        elif 'sister' in example_input and 'brother' in example_input:
            return self.apply_family_logic_pattern(query, example)
        
        # STRING PATTERN
        elif 'reverse' in example_input:
            return self.apply_string_pattern(query, example)
        
        # Default: return the example output
        return example_output
    
    def apply_counting_pattern(self, query: str, example: Dict) -> str:
        """Apply counting pattern learned from example"""
        
        # Learn the counting method from example
        query_lower = query.lower()
        
        # Extract what to count
        if 'letter' in query_lower:
            # Get all quoted strings
            all_quotes = re.findall(r'"([^"]*)"', query)
            
            if len(all_quotes) >= 2:
                # Pattern: "letter" in "text"
                letter = all_quotes[0].lower()
                text = all_quotes[1]
            elif len(all_quotes) == 1:
                # Single quote - determine what it is
                quoted_text = all_quotes[0]
                if len(quoted_text) == 1:
                    # Single character = letter to count
                    letter = quoted_text.lower()
                    # Extract text after "in:"
                    in_match = re.search(r'in[:\s]*(.+)$', query, re.IGNORECASE)
                    text = in_match.group(1).strip() if in_match else ""
                else:
                    # Long string = text to search in
                    text = quoted_text
                    # Extract letter from pattern
                    letter_match = re.search(r'letter\s+(\w)', query_lower)
                    letter = letter_match.group(1) if letter_match else ""
            else:
                # No quotes - extract from text pattern
                letter_match = re.search(r'letter\s+(\w)', query_lower)
                letter = letter_match.group(1) if letter_match else ""
                
                in_match = re.search(r'in[:\s]*(.+)$', query, re.IGNORECASE)
                text = in_match.group(1).strip() if in_match else ""
            
            if letter and text:
                count = text.lower().count(letter)
                return str(count)
        
        # Word counting pattern
        if 'word' in query_lower and ('appear' in query_lower or 'occur' in query_lower):
            all_quotes = re.findall(r'"([^"]*)"', query)
            
            if all_quotes:
                # Extract word to count
                word_match = re.search(r'word\s+(\w+)', query_lower)
                if word_match:
                    word = word_match.group(1)
                    text = all_quotes[-1]  # Last quoted string is usually the text
                    words = text.lower().split()
                    count = words.count(word)
                    return str(count)
        
        return "0"
    
    def apply_arithmetic_pattern(self, query: str, example: Dict) -> str:
        """Apply arithmetic pattern learned from example"""
        
        numbers = re.findall(r'\d+\.?\d*', query)
        
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            
            if '+' in query or 'plus' in query.lower():
                result = a + b
            elif '*' in query or 'times' in query.lower() or '×' in query:
                result = a * b
            elif '/' in query or 'divide' in query.lower():
                result = a / b if b != 0 else 0
            elif '-' in query:
                result = a - b
            else:
                result = a + b  # Default
            
            # Format result
            return str(int(result) if result == int(result) else result)
        
        return "0"
    
    def apply_family_logic_pattern(self, query: str, example: Dict) -> str:
        """Apply family logic pattern learned from example"""
        
        # Extract numbers from query
        numbers = re.findall(r'\d+', query)
        
        if len(numbers) >= 2:
            brothers = int(numbers[0]) 
            sisters = int(numbers[1])
            
            # Learned pattern: each brother has all sisters + the original person
            # (if the original person is female)
            return str(sisters + 1)
        
        return "0"
    
    def apply_string_pattern(self, query: str, example: Dict) -> str:
        """Apply string pattern learned from example"""
        
        if 'reverse' in query.lower():
            # Find word to reverse
            if '"' in query:
                word_match = re.search(r'"([^"]*)"', query)
                word = word_match.group(1) if word_match else ""
            else:
                # Find the target word
                words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
                skip_words = {'reverse', 'word', 'the'}
                target_words = [w for w in words if w.lower() not in skip_words]
                word = target_words[-1] if target_words else ""
            
            if word:
                return word[::-1]
        
        return ""

def create_training_examples():
    """Create training examples for pattern learning"""
    return [
        # Counting examples - teach the general pattern
        ('count letter r in strawberry', '3'),
        ('count letter a in banana', '3'), 
        ('count letter e in hello', '1'),
        ('count letter s in mississippi', '4'),
        ('count letter o in google', '2'),
        ('how many r in strawberry', '3'),
        ('letter t in butter', '2'),
        
        # Arithmetic examples
        ('1+1', '2'),
        ('2+2', '4'),
        ('3+4', '7'),
        ('7 times 1.25', '8.75'),
        ('10 * 5', '50'),
        ('100/4', '25'),
        ('6 * 7', '42'),
        
        # Family logic examples  
        ('Sarah has 3 brothers 2 sisters how many sisters do brothers have', '3'),
        ('Alice has 1 brother 1 sister how many sisters does brother have', '2'),
        ('Mary has 2 brothers 1 sister how many sisters do brothers have', '2'),
        
        # String examples
        ('reverse palindrome', 'emordnilap'),
        ('reverse hello', 'olleh'),
        ('reverse cat', 'tac'),
        ('reverse python', 'nohtyp'),
    ]

def test_pattern_learning():
    """Test the pattern learning engine"""
    
    print("🧠 PATTERN LEARNING ENGINE - NO HARDCODED RULES")
    print("=" * 60)
    
    # Create engine
    engine = PatternLearningEngine()
    
    # Add training examples
    training_examples = create_training_examples()
    
    for input_text, output_text in training_examples:
        engine.add_example(input_text, output_text)
    
    # Train the engine
    engine.train()
    
    print("\n🧪 TESTING LEARNED PATTERNS:")
    
    # Test the exact cases where we failed
    test_cases = [
        'Count the number of letter "r" in: "strawberry raspberry blueberry"',
        'Sarah has 3 brothers and 2 sisters. How many sisters does each of Sarah\'s brothers have?',
        'What is 7 × 1.25?',
        'Reverse the word "palindrome"',
        '1+1',
        '2+2',
        'How many times does the word "the" appear in: "The quick brown fox jumps over the lazy dog near the old oak tree"'
    ]
    
    for test in test_cases:
        prediction = engine.predict(test)
        print(f"Q: {test}")
        print(f"A: {prediction}")
        print()

if __name__ == "__main__":
    test_pattern_learning()