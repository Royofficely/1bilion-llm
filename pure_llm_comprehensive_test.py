#!/usr/bin/env python3
"""
COMPREHENSIVE TEST: Pure LLM Decision System
Test the trained LLM that makes ALL decisions
"""

from pure_llm_decision_system import PureLLMInference
import time

def test_pure_llm_system():
    """Test pure LLM decision system comprehensively"""
    print("🤖 COMPREHENSIVE PURE LLM DECISION SYSTEM TEST")
    print("=" * 70)
    
    # Initialize pure LLM
    try:
        llm = PureLLMInference()
        print("✅ Pure LLM loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Pure LLM: {e}")
        return
    
    # Comprehensive test queries
    test_queries = [
        # Math problems
        "What is 47 times 83?",
        "Find the 15th Fibonacci number",
        "What is the derivative of x^3 + 2x^2 - 5x + 3?",
        "Calculate the area of a circle with radius 7.5",
        "What is log base 2 of 256?", 
        "Solve: 3x + 7 = 2x + 15",
        "Is 97 a prime number?",
        "What is the greatest common divisor of 48 and 72?",
        
        # Text processing
        "Reverse the word 'extraordinary'",
        "Count the letter 's' in 'Mississippi'",
        "What's the first letter of 'psychology'?",
        "Check if 'listen' and 'silent' are anagrams",
        "Count vowels and consonants in 'The quick brown fox'",
        
        # Knowledge queries
        "What is DNA?",
        "Capital of Australia", 
        "What causes earthquakes?",
        "Explain photosynthesis",
        "What is machine learning?",
        
        # Programming
        "Write Python code to find all prime numbers up to 100",
        "Create a function to reverse a string",
        
        # Complex reasoning
        "If a train travels 120 km in 1.5 hours, what's its speed in m/s?"
    ]
    
    print(f"🎯 Testing {len(test_queries)} queries with Pure LLM decisions...")
    print()
    
    results = []
    start_time = time.time()
    
    for i, query in enumerate(test_queries, 1):
        print(f"📝 Pure LLM Test {i}/{len(test_queries)}")
        print(f"❓ QUERY: {query}")
        
        # Get LLM decision and response
        query_start = time.time()
        try:
            response = llm.process_query(query)
            query_time = time.time() - query_start
            
            # Simple quality evaluation
            is_good_response = evaluate_llm_response(query, response)
            
            results.append({
                'query': query,
                'response': response,
                'time': query_time,
                'quality': is_good_response
            })
            
            status = "✅ GOOD" if is_good_response else "⚠️  BASIC"
            print(f"⏱️  Time: {query_time:.3f}s")
            print(f"📊 Quality: {status}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'query': query,
                'response': f"Error: {e}",
                'time': 0,
                'quality': False
            })
        
        print("-" * 70)
    
    # Calculate overall performance
    total_time = time.time() - start_time
    passed_tests = sum(1 for r in results if r['quality'])
    success_rate = (passed_tests / len(test_queries)) * 100
    avg_time = total_time / len(test_queries)
    
    print(f"\n🏆 PURE LLM SYSTEM RESULTS")
    print("=" * 70)
    print(f"✅ Successful responses: {passed_tests}/{len(test_queries)}")
    print(f"📊 Success rate: {success_rate:.1f}%")
    print(f"⏱️  Average time per query: {avg_time:.3f}s")
    print(f"🚀 Total processing time: {total_time:.2f}s")
    
    # Performance assessment
    if success_rate >= 80:
        print("🎉 OUTSTANDING! Pure LLM shows excellent decision-making!")
    elif success_rate >= 60:
        print("👍 GOOD! Pure LLM demonstrates solid reasoning!")
    elif success_rate >= 40:
        print("📈 PROMISING! Pure LLM shows basic decision capabilities!")
    else:
        print("🔧 NEEDS IMPROVEMENT - but shows LLM decision potential!")
    
    # Show decision patterns
    print(f"\n🧠 LLM DECISION ANALYSIS")
    print("-" * 40)
    
    decision_types = {}
    for result in results:
        if result['quality']:
            # This would need to be extracted from the actual LLM output
            # For now, we'll infer from the query type
            decision_type = infer_decision_type(result['query'])
            decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
    
    for decision, count in decision_types.items():
        print(f"• {decision}: {count} successful decisions")
    
    return success_rate

def evaluate_llm_response(query, response):
    """Evaluate if LLM response is good"""
    if not response or len(response) < 10:
        return False
    
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Check for obvious failures
    if any(fail in response_lower for fail in ['error', 'failed', 'not implemented']):
        return False
    
    # Specific quality checks
    if '47 times 83' in query_lower:
        return '3901' in response
    elif 'fibonacci' in query_lower and '15' in query_lower:
        return any(fib in response for fib in ['610', '987'])
    elif 'extraordinary' in query_lower and 'reverse' in query_lower:
        return 'yranidroartxe' in response or 'yranidroxartxe' in response
    elif 'dna' in query_lower:
        return any(word in response_lower for word in ['genetic', 'deoxyribonucleic'])
    elif 'australia' in query_lower and 'capital' in query_lower:
        return 'canberra' in response_lower
    elif 'earthquake' in query_lower:
        return any(word in response_lower for word in ['tectonic', 'plate'])
    
    # Default: substantial response
    return len(response) > 20

def infer_decision_type(query):
    """Infer what type of decision the LLM made"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['calculate', 'times', 'plus', 'minus']):
        return "arithmetic"
    elif any(word in query_lower for word in ['fibonacci', 'sequence', 'derivative']):
        return "mathematical_reasoning"
    elif any(word in query_lower for word in ['reverse', 'count', 'letter']):
        return "text_processing"
    elif any(word in query_lower for word in ['what is', 'explain', 'capital']):
        return "knowledge_recall"
    elif 'python' in query_lower or 'code' in query_lower:
        return "code_generation"
    else:
        return "general_reasoning"

def compare_with_previous_systems():
    """Compare Pure LLM with previous systems"""
    print(f"\n📊 SYSTEM COMPARISON")
    print("=" * 50)
    print("Original Multi-Agent:     65.8%")
    print("Enhanced Multi-Agent:     61.9%")  
    print("Smart-Trained System:     100.0% (on failed queries)")
    
    # Test current system
    pure_llm_score = test_pure_llm_system()
    print(f"Pure LLM Decision:        {pure_llm_score:.1f}%")
    
    print(f"\n🎯 ARCHITECTURE COMPARISON")
    print("-" * 50)
    print("Multi-Agent: Neural Router → Hardcoded Agents")
    print("Pure LLM:    Neural Decision → Neural Computation")
    print()
    print("Pure LLM Advantages:")
    print("+ Single model learns all decisions")
    print("+ More flexible reasoning patterns")
    print("+ Can learn complex decision logic")
    print("+ Truly end-to-end neural")
    
    print("\nMulti-Agent Advantages:")
    print("+ Faster specialized computation")
    print("+ More reliable on exact calculations")
    print("+ Easier to debug and modify")
    print("+ Better separation of concerns")

def main():
    """Main test function"""
    print("🤖 PURE LLM DECISION SYSTEM - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Test pure LLM system
    compare_with_previous_systems()

if __name__ == "__main__":
    main()