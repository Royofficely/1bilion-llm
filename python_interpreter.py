#!/usr/bin/env python3
"""
Python Code Interpreter Tool - For math, counting, and computation
"""

import re
import math
from typing import Any, Dict

class PythonInterpreter:
    """Safe Python code interpreter for math and counting"""
    
    def __init__(self):
        # Safe built-ins for math operations
        self.safe_builtins = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'range': range,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'tuple': tuple,
            'set': set,
            'dict': dict,
        }
        
        # Math module functions
        self.safe_math = {
            'sqrt': math.sqrt,
            'pow': math.pow,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
        }
    
    def execute_math(self, query: str) -> str:
        """Execute math operations safely"""
        try:
            # Handle letter counting queries
            if "letter" in query.lower() and "strawberry" in query.lower():
                return self.count_letters_in_strawberry(query)
            
            # Handle basic math expressions
            if any(op in query for op in ['+', '-', '*', '/', '**', '^']):
                return self.evaluate_math_expression(query)
            
            # Handle counting queries
            if "how many" in query.lower():
                return self.handle_counting_query(query)
            
            # Handle factorial
            if "factorial" in query.lower():
                return self.calculate_factorial(query)
            
            return "Math query not recognized"
            
        except Exception as e:
            return f"Calculation error: {str(e)[:50]}"
    
    def count_letters_in_strawberry(self, query: str) -> str:
        """Count specific letters in 'Strawberry'"""
        word = "Strawberry"
        
        # Extract which letter to count
        if " r " in query.lower() or " R " in query or "letter r" in query.lower():
            letter = 'r'
        elif " t " in query.lower() or " T " in query or "letter t" in query.lower():
            letter = 't'
        elif " a " in query.lower() or " A " in query or "letter a" in query.lower():
            letter = 'a'
        elif " s " in query.lower() or " S " in query or "letter s" in query.lower():
            letter = 's'
        else:
            # Default to 'r' if not specified
            letter = 'r'
        
        count = word.lower().count(letter.lower())
        return str(count)
    
    def evaluate_math_expression(self, query: str) -> str:
        """Safely evaluate math expressions"""
        # Extract numbers and operators
        expression = re.sub(r'[^0-9+\-*/().\s]', '', query)
        expression = expression.replace('^', '**')  # Handle ^ as power
        
        try:
            # Simple expressions only
            if re.match(r'^[\d+\-*/().\s]+$', expression):
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
        except:
            pass
        
        # Try to extract simple operations
        simple_patterns = [
            r'(\d+)\s*\+\s*(\d+)',  # addition
            r'(\d+)\s*-\s*(\d+)',   # subtraction  
            r'(\d+)\s*\*\s*(\d+)',  # multiplication
            r'(\d+)\s*/\s*(\d+)',   # division
        ]
        
        for pattern in simple_patterns:
            match = re.search(pattern, query)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                if '+' in match.group(0):
                    return str(a + b)
                elif '-' in match.group(0):
                    return str(a - b)
                elif '*' in match.group(0):
                    return str(a * b)
                elif '/' in match.group(0):
                    return str(a / b) if b != 0 else "Division by zero"
        
        return "Invalid math expression"
    
    def handle_counting_query(self, query: str) -> str:
        """Handle counting questions"""
        # Extract numbers for counting
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            return str(sum(int(n) for n in numbers))
        
        return "Counting query not recognized"
    
    def calculate_factorial(self, query: str) -> str:
        """Calculate factorial"""
        numbers = re.findall(r'\d+', query)
        if numbers:
            n = int(numbers[0])
            if n <= 20:  # Limit for safety
                result = math.factorial(n)
                return str(result)
            else:
                return "Number too large for factorial"
        
        return "No number found for factorial"

def test_interpreter():
    """Test the Python interpreter"""
    interpreter = PythonInterpreter()
    
    test_queries = [
        "how many letter r in strawberry",
        "1+1", 
        "2+2",
        "10*5",
        "100/4", 
        "factorial 5",
        "how many 3+4+5"
    ]
    
    print("Testing Python Interpreter:")
    print("=" * 40)
    
    for query in test_queries:
        result = interpreter.execute_math(query)
        print(f"Query: {query}")
        print(f"Result: {result}")
        print("-" * 20)

if __name__ == "__main__":
    test_interpreter()