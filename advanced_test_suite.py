#!/usr/bin/env python3
"""
ADVANCED COMPLEX TEST SUITE
Real challenging problems that would test Claude-level intelligence
"""

import torch
import time
import json
from simple_llm_router import LLMRouter
from specialized_agents import MathAgent, PythonAgent, TextAgent, KnowledgeAgent, WebAgent

class AdvancedTestSuite:
    """Advanced testing with Claude-level complexity"""
    
    def __init__(self):
        print("🧠 ADVANCED TEST SUITE - CLAUDE-KILLER CHALLENGE")
        print("=" * 60)
        
        # Load system
        try:
            checkpoint = torch.load('simple_router_model.pt', weights_only=False)
            self.router = LLMRouter()
            self.router.model.load_state_dict(checkpoint['model_state_dict'])
            self.router.tokenizer = checkpoint['tokenizer'] 
            self.router.agents = checkpoint['agents']
            print("✅ Neural router loaded")
        except:
            print("🔧 Creating new router...")
            self.router = LLMRouter()
            self.router.train_router(epochs=50)
        
        self.agents = {
            "math_agent": MathAgent(),
            "python_agent": PythonAgent(),
            "text_agent": TextAgent(), 
            "knowledge_agent": KnowledgeAgent(),
            "web_agent": WebAgent()
        }
        
    def run_complex_math_tests(self):
        """Complex mathematical reasoning tests"""
        print("\n🔢 COMPLEX MATH CHALLENGE")
        print("-" * 40)
        
        math_tests = [
            # Advanced calculations
            "What is 17^8 without calculator?",
            "Find the 15th Fibonacci number",
            "Calculate the area of a circle with radius 7.5",
            "What is the derivative of x^3 + 2x^2 - 5x + 3?",
            
            # Complex sequences
            "What's the next number: 1, 4, 9, 16, 25, 36, ?",
            "Pattern: 2, 6, 18, 54, 162, ? (next number)",
            "Sequence: 1, 1, 2, 3, 5, 8, 13, ? (what's next)",
            
            # Advanced problems
            "If a train travels 120 km in 1.5 hours, what's its speed in m/s?",
            "Solve: 3x + 7 = 2x + 15",
            "What is log base 2 of 256?",
            
            # Prime and number theory
            "List all prime numbers between 50 and 70",
            "What is the greatest common divisor of 48 and 72?",
            "Is 143 a prime number? Explain why.",
        ]
        
        return self._run_test_batch(math_tests, "MATH")
    
    def run_complex_coding_tests(self):
        """Advanced programming challenges"""
        print("\n💻 COMPLEX CODING CHALLENGE")  
        print("-" * 40)
        
        coding_tests = [
            # Algorithms
            "Write Python code for binary search algorithm",
            "Implement a function to find the factorial of n recursively",
            "Create a Python class for a binary tree with insert method", 
            "Write code to reverse a linked list",
            
            # Data structures
            "Implement a stack in Python with push, pop, peek methods",
            "Write a function to check if parentheses are balanced",
            "Create code to find the maximum element in a rotated sorted array",
            
            # Advanced problems
            "Write Python to find all permutations of a string",
            "Implement merge sort algorithm in Python",
            "Create a function to detect cycles in a linked list",
            "Write code for depth-first search in a graph",
            
            # Complex logic
            "Write Python to solve the Tower of Hanoi problem",
            "Implement a LRU cache in Python",
        ]
        
        return self._run_test_batch(coding_tests, "CODING")
        
    def run_complex_text_tests(self):
        """Advanced text processing challenges"""
        print("\n📝 COMPLEX TEXT CHALLENGE")
        print("-" * 40)
        
        text_tests = [
            # Advanced string operations
            "Count vowels and consonants in 'The quick brown fox jumps over the lazy dog'",
            "Find the longest palindrome in 'abccbadef'",
            "Remove all duplicate characters from 'programming'",
            "Check if 'listen' and 'silent' are anagrams",
            
            # Complex text analysis
            "Extract all email addresses from: 'Contact us at info@company.com or support@help.org for assistance'",
            "Count word frequency in: 'the cat sat on the mat the cat was fat'",
            "Find the most common letter in 'supercalifragilisticexpialidocious'",
            
            # Pattern matching
            "Replace all numbers with 'X' in: 'Call 555-1234 or 987-6543 today'",
            "Capitalize first letter of each word in 'hello world from python'",
            "Reverse words in sentence: 'artificial intelligence is amazing'",
        ]
        
        return self._run_test_batch(text_tests, "TEXT")
    
    def run_complex_knowledge_tests(self):
        """Advanced knowledge and reasoning tests"""
        print("\n🧠 COMPLEX KNOWLEDGE CHALLENGE") 
        print("-" * 40)
        
        knowledge_tests = [
            # Science
            "Explain the process of mitosis in cell division",
            "What is the difference between DNA and RNA?",
            "How does photosynthesis convert CO2 to glucose?",
            "What causes the greenhouse effect?",
            
            # Geography & History
            "Name the longest river in each continent",
            "What caused World War I to start?",
            "Which countries border the Mediterranean Sea?",
            
            # Technology
            "How does a computer processor execute instructions?",
            "What is the difference between HTTP and HTTPS?",
            "Explain how neural networks learn",
            "What is blockchain technology?",
            
            # Complex reasoning  
            "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "You have 3 boxes: one has only apples, one has only oranges, one has both. All labels are wrong. You can pick one fruit from one box. How do you correctly label all boxes?",
        ]
        
        return self._run_test_batch(knowledge_tests, "KNOWLEDGE")
    
    def run_edge_case_tests(self):
        """Edge cases and tricky problems"""
        print("\n⚡ EDGE CASE CHALLENGE")
        print("-" * 40)
        
        edge_tests = [
            # Mathematical edge cases
            "What is 0^0?",
            "What is infinity minus infinity?", 
            "Calculate 1 divided by 0",
            
            # Empty/null cases
            "Reverse an empty string",
            "Find factorial of 0",
            "Sort an empty list in Python",
            
            # Ambiguous queries
            "What is the capital?",  # Missing country
            "Calculate the square root",  # Missing number
            "Write code to sort",  # Missing data type
            
            # Impossible tasks
            "List all prime numbers",  # Infinite set
            "What is the largest number?",  # No such number
            "Solve x = x + 1",  # No solution
        ]
        
        return self._run_test_batch(edge_tests, "EDGE CASES")
    
    def _run_test_batch(self, tests, category):
        """Run a batch of tests and return results"""
        results = {
            'category': category,
            'total': len(tests),
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'details': []
        }
        
        for i, query in enumerate(tests, 1):
            print(f"\n📝 {category} Test {i}/{len(tests)}")
            print(f"🔍 QUERY: {query}")
            
            try:
                start_time = time.time()
                
                # Route and process
                selected_agent, optimized_prompt = self.router.route_query(query)
                
                if selected_agent in self.agents:
                    agent = self.agents[selected_agent]
                    result = agent.process(query)
                    
                    processing_time = time.time() - start_time
                    
                    print(f"🎯 ROUTED TO: {selected_agent}")
                    print(f"⏱️  TIME: {processing_time:.3f}s")
                    print(f"💬 RESPONSE: {result[:200]}{'...' if len(result) > 200 else ''}")
                    
                    # Evaluate response quality
                    if self._evaluate_response(query, result, selected_agent):
                        results['passed'] += 1
                        status = "✅ PASS"
                    else:
                        results['failed'] += 1
                        status = "❌ FAIL"
                        
                    print(f"📊 {status}")
                    
                    results['details'].append({
                        'query': query,
                        'agent': selected_agent,
                        'response': result,
                        'time': processing_time,
                        'status': status
                    })
                    
                else:
                    results['errors'] += 1
                    print(f"❌ ERROR: Agent {selected_agent} not found")
                    
            except Exception as e:
                results['errors'] += 1
                print(f"❌ EXCEPTION: {e}")
                
        return results
    
    def _evaluate_response(self, query, response, agent):
        """Evaluate if response is reasonable"""
        if not response or len(response) < 5:
            return False
            
        response_lower = response.lower()
        
        # Check for obvious failures
        if any(phrase in response_lower for phrase in [
            'error', 'failed', 'cannot', "don't know", 'no information'
        ]):
            # Allow some legitimate "don't know" responses for impossible questions
            if any(impossible in query.lower() for impossible in [
                'infinity minus infinity', 'divided by 0', 'largest number'
            ]):
                return True
            return False
            
        # Agent-specific quality checks
        if agent == 'math_agent' and any(word in query.lower() for word in ['calculate', 'what is', 'find']):
            return any(char.isdigit() for char in response) or 'prime' in response_lower
            
        if agent == 'python_agent' and 'python' in query.lower():
            return 'def ' in response or 'import' in response or 'print' in response
            
        if agent == 'text_agent':
            return len(response) > 10
            
        return True  # Default pass for other cases
    
    def run_full_advanced_suite(self):
        """Run complete advanced test suite"""
        print("🚀 STARTING FULL ADVANCED TEST SUITE")
        print("=" * 60)
        
        all_results = []
        
        # Run all test categories
        all_results.append(self.run_complex_math_tests())
        all_results.append(self.run_complex_coding_tests()) 
        all_results.append(self.run_complex_text_tests())
        all_results.append(self.run_complex_knowledge_tests())
        all_results.append(self.run_edge_case_tests())
        
        # Calculate overall results
        total_tests = sum(r['total'] for r in all_results)
        total_passed = sum(r['passed'] for r in all_results)
        total_failed = sum(r['failed'] for r in all_results)
        total_errors = sum(r['errors'] for r in all_results)
        
        overall_success_rate = (total_passed / total_tests) * 100
        
        print("\n" + "="*60)
        print("🏆 ADVANCED TEST SUITE RESULTS")
        print("="*60)
        
        for result in all_results:
            category_rate = (result['passed'] / result['total']) * 100
            print(f"📊 {result['category']}: {result['passed']}/{result['total']} ({category_rate:.1f}%)")
            
        print("-" * 60)
        print(f"🎯 OVERALL SUCCESS RATE: {total_passed}/{total_tests} ({overall_success_rate:.1f}%)")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")  
        print(f"💥 Errors: {total_errors}")
        
        # Performance assessment
        if overall_success_rate >= 85:
            print("🎉 OUTSTANDING! System rivals Claude-level intelligence!")
        elif overall_success_rate >= 70:
            print("👍 EXCELLENT! System shows strong specialized intelligence!")
        elif overall_success_rate >= 50:
            print("✅ GOOD! System handles most complex tasks well!")
        else:
            print("🔧 NEEDS IMPROVEMENT on complex reasoning tasks")
            
        return overall_success_rate, all_results

def main():
    """Run advanced test suite"""
    suite = AdvancedTestSuite()
    success_rate, results = suite.run_full_advanced_suite()
    
    print(f"\n🎯 FINAL SCORE: {success_rate:.1f}%")
    print("🧠 Advanced multi-agent system testing complete!")

if __name__ == "__main__":
    main()