#!/usr/bin/env python3
"""
Local test script for GPT Killer V2
"""

from gpt_killer_v2 import GPTKillerV2

def test_gpt_killer_v2():
    """Test GPT Killer V2 locally with various queries"""
    
    ai = GPTKillerV2()
    
    test_queries = [
        # Math and counting tests
        "how many times letter R appears in Strawberry",
        "how many letter r in strawberry", 
        "1+1",
        "2+2",
        "10*5",
        "100/4",
        "calculate factorial 5",
        "what is 15 + 25",
        
        # Web search tests
        "who is Roy Nativ",
        "bitcoin price",
        "time in bangkok",
        "date in israel", 
        "news of israel today",
        "weather today",
        
        # Direct answer tests
        "hey",
        "hello", 
        "thanks",
        
        # Mixed tests
        "what is the weather",
        "current events",
        "compute 200 / 8"
    ]
    
    print("🚀 GPT KILLER V2 LOCAL TEST")
    print("=" * 60)
    
    success_count = 0
    total_count = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        try:
            # Show routing decision
            endpoint = ai.router.route_query(query)
            response = ai.get_response(query)
            
            print(f"[{i:2d}/{total_count}] {query}")
            print(f"      Route: {endpoint}")
            print(f"      Answer: {response}")
            
            # Simple success check (not empty, not "Search failed")
            if response and response != "Search failed" and "Error" not in response:
                success_count += 1
                print("      ✅ SUCCESS")
            else:
                print("      ❌ FAILED")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"      ❌ ERROR: {e}")
            print("-" * 50)
    
    # Summary
    success_rate = (success_count / total_count) * 100
    print(f"\n🎯 RESULTS: {success_count}/{total_count} successful ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🏆 EXCELLENT - Ready for production!")
    elif success_rate >= 75:
        print("✅ GOOD - Minor tweaks needed")
    elif success_rate >= 50:
        print("⚠️  NEEDS WORK - Major improvements required")
    else:
        print("❌ POOR - Significant fixes needed")

if __name__ == "__main__":
    test_gpt_killer_v2()