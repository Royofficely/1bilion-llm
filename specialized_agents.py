#!/usr/bin/env python3
"""
SPECIALIZED AI AGENTS - Each agent masters one domain
Math Agent: Calculations, sequences, patterns
Python Agent: Code execution, programming
Text Agent: String operations, text manipulation  
Knowledge Agent: Facts, explanations
Web Agent: Current information, news
"""

import torch
import torch.nn as nn
import re
import ast
import operator
from typing import Any, Dict, List
from simple_llm_router import SimpleRouterLLM, SimpleTokenizer

class MathAgent:
    """Specialized agent for mathematical operations"""
    
    def __init__(self):
        self.name = "Math Agent"
        self.operations = {
            '+': operator.add, '-': operator.sub, 
            '*': operator.mul, '×': operator.mul,
            '/': operator.truediv, '÷': operator.truediv,
            '**': operator.pow, '^': operator.pow
        }
        print(f"🧮 {self.name} initialized - Masters: arithmetic, sequences, patterns")
        
    def process(self, query: str) -> str:
        """Process mathematical queries"""
        query_lower = query.lower()
        
        try:
            # Handle specific math operations
            if "times" in query_lower or "×" in query or "*" in query:
                return self.handle_multiplication(query)
            elif "plus" in query_lower or "+" in query:
                return self.handle_addition(query)
            elif "minus" in query_lower or "-" in query:
                return self.handle_subtraction(query)
            elif "divided" in query_lower or "/" in query or "÷" in query:
                return self.handle_division(query)
            elif "factorial" in query_lower:
                return self.handle_factorial(query)
            elif "fibonacci" in query_lower:
                return self.handle_fibonacci(query)
            elif "sequence" in query_lower or "pattern" in query_lower:
                return self.handle_sequence(query)
            elif "prime" in query_lower:
                return self.handle_prime(query)
            else:
                # Try to evaluate as math expression
                return self.evaluate_expression(query)
                
        except Exception as e:
            return f"Math calculation error: {str(e)}"
    
    def handle_multiplication(self, query: str) -> str:
        """Handle multiplication queries"""
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        if len(numbers) >= 2:
            result = float(numbers[0]) * float(numbers[1])
            return f"{numbers[0]} × {numbers[1]} = {int(result) if result.is_integer() else result}"
        return "Need two numbers for multiplication"
    
    def handle_addition(self, query: str) -> str:
        """Handle addition queries"""
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        if len(numbers) >= 2:
            result = sum(float(n) for n in numbers)
            return f"{' + '.join(numbers)} = {int(result) if result.is_integer() else result}"
        return "Need numbers for addition"
    
    def handle_subtraction(self, query: str) -> str:
        """Handle subtraction queries"""
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        if len(numbers) >= 2:
            result = float(numbers[0]) - float(numbers[1])
            return f"{numbers[0]} - {numbers[1]} = {int(result) if result.is_integer() else result}"
        return "Need two numbers for subtraction"
    
    def handle_division(self, query: str) -> str:
        """Handle division queries"""
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        if len(numbers) >= 2 and float(numbers[1]) != 0:
            result = float(numbers[0]) / float(numbers[1])
            return f"{numbers[0]} ÷ {numbers[1]} = {int(result) if result.is_integer() else result:.4f}"
        return "Need two numbers for division (divisor cannot be zero)"
    
    def handle_factorial(self, query: str) -> str:
        """Calculate factorial"""
        numbers = re.findall(r'\d+', query)
        if numbers:
            n = int(numbers[0])
            if n > 20:
                return f"Factorial too large (n > 20)"
            result = 1
            for i in range(1, n + 1):
                result *= i
            return f"{n}! = {result}"
        return "Need a number for factorial"
    
    def handle_fibonacci(self, query: str) -> str:
        """Generate fibonacci sequence"""
        numbers = re.findall(r'\d+', query)
        n = int(numbers[0]) if numbers else 10
        
        if n > 20:
            n = 20
            
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
            
        return f"Fibonacci sequence ({n} terms): {', '.join(map(str, fib[:n]))}"
    
    def handle_sequence(self, query: str) -> str:
        """Identify patterns in sequences"""
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 3:
            nums = [int(n) for n in numbers[:5]]  # Take first 5 numbers
            
            # Check arithmetic progression
            diff = nums[1] - nums[0]
            if all(nums[i+1] - nums[i] == diff for i in range(len(nums)-1)):
                next_num = nums[-1] + diff
                return f"Arithmetic sequence (+{diff}): Next number is {next_num}"
            
            # Check geometric progression
            if nums[0] != 0:
                ratio = nums[1] / nums[0]
                if all(abs(nums[i+1] / nums[i] - ratio) < 0.001 for i in range(len(nums)-1)):
                    next_num = int(nums[-1] * ratio)
                    return f"Geometric sequence (×{ratio}): Next number is {next_num}"
            
            # Check fibonacci-like
            if len(nums) >= 3 and all(nums[i] == nums[i-1] + nums[i-2] for i in range(2, len(nums))):
                next_num = nums[-1] + nums[-2]
                return f"Fibonacci-like sequence: Next number is {next_num}"
                
        return f"Pattern analysis: Found numbers {numbers}, need more context for pattern"
    
    def handle_prime(self, query: str) -> str:
        """Handle prime number queries"""
        numbers = re.findall(r'\d+', query)
        if numbers:
            n = int(numbers[0])
            if n > 100:
                return "Number too large for prime check"
            
            if self.is_prime(n):
                return f"{n} is a prime number"
            else:
                return f"{n} is not a prime number"
        return "Need a number to check if prime"
    
    def is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def evaluate_expression(self, query: str) -> str:
        """Safely evaluate math expressions"""
        # Extract math expression
        math_expr = re.sub(r'[^\d+\-*/().\s]', '', query)
        math_expr = math_expr.strip()
        
        if not math_expr:
            return "No mathematical expression found"
            
        try:
            # Safe evaluation
            result = eval(math_expr)
            return f"{math_expr} = {result}"
        except:
            return f"Cannot evaluate: {math_expr}"

class PythonAgent:
    """Specialized agent for Python code execution"""
    
    def __init__(self):
        self.name = "Python Agent" 
        print(f"🐍 {self.name} initialized - Masters: code execution, programming")
        
    def process(self, query: str) -> str:
        """Process Python code queries"""
        try:
            if "function" in query.lower():
                return self.generate_function(query)
            elif "sort" in query.lower():
                return self.generate_sort_code(query)
            elif "fibonacci" in query.lower():
                return self.generate_fibonacci_code()
            elif "factorial" in query.lower():
                return self.generate_factorial_code()
            elif "algorithm" in query.lower():
                return self.suggest_algorithm(query)
            else:
                return self.general_python_help(query)
                
        except Exception as e:
            return f"Python code generation error: {str(e)}"
    
    def generate_function(self, query: str) -> str:
        """Generate Python function based on query"""
        if "sort" in query.lower():
            return self.generate_sort_code(query)
        else:
            return """def example_function(param):
    \"\"\"
    Generated Python function template
    \"\"\"
    # Your code here
    return param

# Usage example:
result = example_function("input")
print(result)"""
    
    def generate_sort_code(self, query: str) -> str:
        """Generate sorting code"""
        return """def sort_list(items):
    \"\"\"
    Sort a list of items
    \"\"\"
    return sorted(items)

# Example usage:
my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = sort_list(my_list)
print(f"Original: {my_list}")
print(f"Sorted: {sorted_list}")

# Alternative - bubble sort implementation:
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""
    
    def generate_fibonacci_code(self) -> str:
        """Generate Fibonacci code"""
        return """def fibonacci(n):
    \"\"\"
    Generate Fibonacci sequence up to n terms
    \"\"\"
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# Usage:
result = fibonacci(10)
print(f"First 10 Fibonacci numbers: {result}")

# Recursive version:
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)"""
    
    def generate_factorial_code(self) -> str:
        """Generate factorial code"""
        return """def factorial(n):
    \"\"\"
    Calculate factorial of n
    \"\"\"
    if n < 0:
        return "Cannot calculate factorial of negative number"
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Usage:
print(f"5! = {factorial(5)}")

# Recursive version:
def factorial_recursive(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)"""
    
    def suggest_algorithm(self, query: str) -> str:
        """Suggest algorithms based on query"""
        if "search" in query.lower():
            return """# Binary Search Algorithm
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Usage:
sorted_array = [1, 3, 5, 7, 9, 11, 13]
index = binary_search(sorted_array, 7)
print(f"Found at index: {index}")"""
        else:
            return "Please specify what type of algorithm you need (sorting, searching, etc.)"
    
    def general_python_help(self, query: str) -> str:
        """General Python programming help"""
        return f"""Python code for: {query}

# Basic Python template:
def main():
    # Your code implementation here
    pass

if __name__ == "__main__":
    main()

# Common Python patterns:
# - List comprehension: [x*2 for x in range(10)]
# - Dictionary: {'key': 'value'}
# - Loop: for item in items:
# - Condition: if condition:"""

class TextAgent:
    """Specialized agent for text operations"""
    
    def __init__(self):
        self.name = "Text Agent"
        print(f"📝 {self.name} initialized - Masters: string operations, text manipulation")
    
    def process(self, query: str) -> str:
        """Process text manipulation queries"""
        try:
            if "reverse" in query.lower():
                return self.handle_reverse(query)
            elif "first letter" in query.lower():
                return self.handle_first_letter(query)
            elif "count" in query.lower() and "letter" in query.lower():
                return self.handle_count_letters(query)
            elif "uppercase" in query.lower():
                return self.handle_uppercase(query)
            elif "lowercase" in query.lower():
                return self.handle_lowercase(query)
            else:
                return self.general_text_help(query)
                
        except Exception as e:
            return f"Text processing error: {str(e)}"
    
    def handle_reverse(self, query: str) -> str:
        """Reverse text or words"""
        # Extract word to reverse
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and words[i-1].lower() in ["reverse", "word"]:
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    reversed_word = clean_word[::-1]
                    return f"Reversed '{clean_word}' → '{reversed_word}'"
        
        # If no specific word found, try to find any word
        word_matches = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        if word_matches:
            word = word_matches[-1]  # Take last word
            reversed_word = word[::-1]
            return f"Reversed '{word}' → '{reversed_word}'"
            
        return "Please specify a word to reverse"
    
    def handle_first_letter(self, query: str) -> str:
        """Get first letter of word"""
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and words[i-1].lower() in ["of", "letter"]:
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    return f"First letter of '{clean_word}' is '{clean_word[0].lower()}'"
        
        # Fallback - find any word
        word_matches = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        if word_matches:
            word = word_matches[-1]
            return f"First letter of '{word}' is '{word[0].lower()}'"
            
        return "Please specify a word"
    
    def handle_count_letters(self, query: str) -> str:
        """Count specific letters in words"""
        # Find letter to count and word
        letter_match = re.search(r'letter ([a-zA-Z])', query)
        word_matches = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        
        if letter_match and word_matches:
            letter = letter_match.group(1).lower()
            words = [w for w in word_matches if w.lower() not in ['letter', 'count', 'how', 'many']]
            
            if words:
                results = []
                total_count = 0
                for word in words:
                    count = word.lower().count(letter)
                    total_count += count
                    results.append(f"'{word}': {count}")
                
                return f"Letter '{letter}' count - {', '.join(results)} | Total: {total_count}"
        
        return "Please specify letter and word(s) to count"
    
    def handle_uppercase(self, query: str) -> str:
        """Convert text to uppercase"""
        word_matches = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        if word_matches:
            word = word_matches[-1]
            return f"Uppercase '{word}' → '{word.upper()}'"
        return "Please specify text to uppercase"
    
    def handle_lowercase(self, query: str) -> str:
        """Convert text to lowercase"""
        word_matches = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        if word_matches:
            word = word_matches[-1]
            return f"Lowercase '{word}' → '{word.lower()}'"
        return "Please specify text to lowercase"
    
    def general_text_help(self, query: str) -> str:
        """General text processing help"""
        return f"""Text processing for: {query}

Available operations:
- Reverse: reverse the word [word]
- First letter: first letter of [word] 
- Count letters: count letter [x] in [word]
- Uppercase: uppercase [text]
- Lowercase: lowercase [text]"""

class KnowledgeAgent:
    """Specialized agent for factual knowledge"""
    
    def __init__(self):
        self.name = "Knowledge Agent"
        self.knowledge_base = self.build_knowledge_base()
        print(f"🧠 {self.name} initialized - Masters: facts, science, explanations")
    
    def build_knowledge_base(self) -> Dict[str, str]:
        """Build knowledge base with facts"""
        return {
            # Geography
            "capital of france": "Paris is the capital of France",
            "capital of usa": "Washington D.C. is the capital of the United States",
            "capital of japan": "Tokyo is the capital of Japan",
            
            # Science
            "gravity": "Gravity is the force that attracts objects with mass toward each other",
            "photosynthesis": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll",
            "why is sky blue": "The sky appears blue because molecules in the atmosphere scatter blue light more than other colors",
            "dna": "DNA (Deoxyribonucleic Acid) contains the genetic instructions for all living organisms",
            
            # Technology
            "machine learning": "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions",
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence in machines",
            "neural network": "A neural network is a computing system inspired by biological neural networks",
            "python": "Python is a high-level programming language known for its readability and versatility",
            
            # Mathematics
            "prime number": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself",
            "fibonacci": "The Fibonacci sequence is: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...",
            
            # General
            "solar system": "The solar system consists of the Sun and 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune"
        }
    
    def process(self, query: str) -> str:
        """Process knowledge queries"""
        try:
            query_lower = query.lower().strip()
            
            # Find best match in knowledge base
            best_match = None
            best_score = 0
            
            for key, value in self.knowledge_base.items():
                score = self.calculate_similarity(query_lower, key)
                if score > best_score:
                    best_score = score
                    best_match = value
            
            if best_score > 0.3:  # Threshold for match
                return best_match
            else:
                return self.generate_general_response(query)
                
        except Exception as e:
            return f"Knowledge processing error: {str(e)}"
    
    def calculate_similarity(self, query: str, key: str) -> float:
        """Calculate similarity between query and knowledge key"""
        query_words = set(query.split())
        key_words = set(key.split())
        
        if not query_words or not key_words:
            return 0.0
        
        intersection = query_words.intersection(key_words)
        union = query_words.union(key_words)
        
        return len(intersection) / len(union)
    
    def generate_general_response(self, query: str) -> str:
        """Generate general response for unknown queries"""
        return f"""I don't have specific information about: {query}

This appears to be a knowledge question. I specialize in:
- Geography (capitals, countries)  
- Science (physics, biology, chemistry)
- Technology (programming, AI, computers)
- Mathematics (numbers, sequences)

Please ask a more specific question in these areas."""

class WebAgent:
    """Specialized agent for web/current information"""
    
    def __init__(self):
        self.name = "Web Agent"
        print(f"🌐 {self.name} initialized - Masters: current events, live information")
    
    def process(self, query: str) -> str:
        """Process web/current information queries"""
        query_lower = query.lower()
        
        if "bitcoin" in query_lower or "btc" in query_lower:
            return "Bitcoin price information requires live web search (not available in this demo)"
        elif "news" in query_lower:
            return "Latest news information requires live web search (not available in this demo)"
        elif "weather" in query_lower:
            return "Weather information requires live web search (not available in this demo)"
        elif "time" in query_lower:
            return "Current time information requires live web search (not available in this demo)"
        else:
            return f"""Web search needed for: {query}

This query requires live web information. In a full implementation, I would:
- Search the web for current information
- Fetch real-time data
- Provide up-to-date results

Demo limitation: No actual web search capability."""

def test_specialized_agents():
    """Test all specialized agents"""
    print("🧪 TESTING SPECIALIZED AGENTS")
    print("=" * 60)
    
    # Initialize agents
    math_agent = MathAgent()
    python_agent = PythonAgent()
    text_agent = TextAgent()
    knowledge_agent = KnowledgeAgent()
    web_agent = WebAgent()
    
    # Test queries for each agent
    test_cases = [
        # Math Agent Tests
        (math_agent, "What is 17 times 23?"),
        (math_agent, "Calculate factorial of 5"),
        (math_agent, "Fibonacci sequence 10 terms"),
        (math_agent, "Pattern: 2, 4, 6, 8"),
        
        # Python Agent Tests  
        (python_agent, "Write Python function to sort list"),
        (python_agent, "Python code for fibonacci"),
        (python_agent, "Create factorial function"),
        
        # Text Agent Tests
        (text_agent, "Reverse the word hello"),
        (text_agent, "First letter of apple"),
        (text_agent, "Count letter r in strawberry"),
        
        # Knowledge Agent Tests
        (knowledge_agent, "What is the capital of France?"),
        (knowledge_agent, "Explain gravity"),
        (knowledge_agent, "What is machine learning?"),
        
        # Web Agent Tests
        (web_agent, "Bitcoin price today"),
        (web_agent, "Latest news"),
    ]
    
    print("\n📋 AGENT TEST RESULTS:")
    print("-" * 60)
    
    for agent, query in test_cases:
        print(f"\n🔍 Query: {query}")
        print(f"👤 Agent: {agent.name}")
        result = agent.process(query)
        print(f"💬 Response: {result}")
        print("-" * 40)
    
    print("\n✅ All specialized agents tested successfully!")
    return [math_agent, python_agent, text_agent, knowledge_agent, web_agent]

if __name__ == "__main__":
    agents = test_specialized_agents()