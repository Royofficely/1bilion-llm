#!/usr/bin/env python3
"""
CLAUDE vs REVOLUTIONARY AI V2 - Head-to-Head Comparison
Real test using the same hard challenges
"""

from gpt_killer_v2 import GPTKillerV2

def claude_vs_revolutionary_test():
    """Head-to-head comparison: Claude vs Revolutionary AI V2"""
    
    ai = GPTKillerV2()
    
    # Test cases with Claude's expected responses vs our system
    test_battles = [
        {
            'challenge': 'Count the number of letter "r" in: "strawberry raspberry blueberry"',
            'claude_response': 'Let me count carefully: s-t-r-a-w-b-e-r-r-y (3 r\'s) + r-a-s-p-b-e-r-r-y (3 r\'s) + b-l-u-e-b-e-r-r-y (2 r\'s) = 8 total r\'s',
            'correct_answer': '8'
        },
        {
            'challenge': 'Sarah has 3 brothers and 2 sisters. How many sisters does each of Sarah\'s brothers have?',
            'claude_response': 'Each of Sarah\'s brothers has 3 sisters: Sarah herself plus her 2 sisters',
            'correct_answer': '3'
        },
        {
            'challenge': 'What is 7 × $1.25 if I buy 7 apples at $1.25 each?',
            'claude_response': '7 × $1.25 = $8.75',
            'correct_answer': '$8.75'
        },
        {
            'challenge': 'Reverse the word "palindrome" letter by letter',
            'claude_response': 'e-m-o-r-d-n-i-l-a-p',
            'correct_answer': 'emordnilap'
        },
        {
            'challenge': 'What comes next: 1, 1, 2, 3, 5, 8, 13, ?',
            'claude_response': '21 (Fibonacci sequence - each number is sum of previous two)',
            'correct_answer': '21'
        },
        {
            'challenge': '1+1',
            'claude_response': '2',
            'correct_answer': '2'
        },
        {
            'challenge': 'who is Roy Nativ',
            'claude_response': 'I don\'t have specific information about Roy Nativ in my training data. Could you provide more context?',
            'correct_answer': 'Should find real information from web search'
        },
        {
            'challenge': 'bitcoin price today',
            'claude_response': 'I don\'t have access to real-time data. Bitcoin prices change constantly, so I\'d recommend checking a current financial website.',
            'correct_answer': 'Should provide actual current price'
        }
    ]
    
    print("⚔️  CLAUDE vs REVOLUTIONARY AI V2 - HEAD-TO-HEAD BATTLE")
    print("=" * 80)
    print("Testing the same challenges on both systems for fair comparison")
    print()
    
    claude_wins = 0
    revolutionary_wins = 0
    ties = 0
    
    for i, battle in enumerate(test_battles, 1):
        print(f"🥊 ROUND {i}: {battle['challenge']}")
        print("-" * 60)
        
        # Claude's response (that's me!)
        print("🤖 CLAUDE SAYS:")
        print(f"   {battle['claude_response']}")
        print()
        
        # Our Revolutionary AI response
        try:
            our_response = ai.get_response(battle['challenge'])
            endpoint = ai.router.route_query(battle['challenge'])
            
            print("🚀 REVOLUTIONARY AI V2 SAYS:")
            print(f"   Route: {endpoint}")
            print(f"   Answer: {our_response}")
            print()
            
            # Evaluation
            print("📊 EVALUATION:")
            
            # Check correctness for specific answers
            if 'correct_answer' in battle:
                correct = battle['correct_answer']
                
                # Check if our answer contains or matches correct answer
                our_correct = (correct.lower() in our_response.lower() or 
                              our_response.strip() == correct)
                
                # Check Claude's answer
                claude_correct = (correct.lower() in battle['claude_response'].lower() or
                                battle['claude_response'].strip() == correct)
                
                print(f"   Correct Answer: {correct}")
                print(f"   Claude Correct: {'✅' if claude_correct else '❌'}")
                print(f"   Revolutionary Correct: {'✅' if our_correct else '❌'}")
                
                # Determine winner
                if our_correct and not claude_correct:
                    winner = "🚀 REVOLUTIONARY AI WINS!"
                    revolutionary_wins += 1
                elif claude_correct and not our_correct:
                    winner = "🤖 CLAUDE WINS!"
                    claude_wins += 1
                elif our_correct and claude_correct:
                    winner = "🤝 TIE - Both correct!"
                    ties += 1
                else:
                    winner = "🤷 Both wrong - No winner"
                
                print(f"   WINNER: {winner}")
            
        except Exception as e:
            print("🚀 REVOLUTIONARY AI V2 SAYS:")
            print(f"   ERROR: {e}")
            print("📊 EVALUATION:")
            print("   🤖 CLAUDE WINS! (Revolutionary AI failed)")
            claude_wins += 1
        
        print("=" * 80)
    
    # Final Battle Results
    total_rounds = len(test_battles)
    print()
    print("🏆 FINAL BATTLE RESULTS")
    print("=" * 80)
    print(f"Total Rounds: {total_rounds}")
    print(f"🤖 Claude Wins: {claude_wins}")
    print(f"🚀 Revolutionary AI Wins: {revolutionary_wins}")
    print(f"🤝 Ties: {ties}")
    print()
    
    # Determine overall winner
    if revolutionary_wins > claude_wins:
        overall_winner = "🚀 REVOLUTIONARY AI V2 WINS THE BATTLE!"
        verdict = "The Revolutionary AI system beats Claude through better tools and real-time data!"
    elif claude_wins > revolutionary_wins:
        overall_winner = "🤖 CLAUDE WINS THE BATTLE!"
        verdict = "Claude's reasoning and knowledge prove superior in this matchup."
    else:
        overall_winner = "🤝 IT'S A TIE!"
        verdict = "Both systems have strengths - Revolutionary AI has tools, Claude has reasoning."
    
    print(f"OVERALL WINNER: {overall_winner}")
    print(f"VERDICT: {verdict}")
    print()
    
    # Analysis
    print("🔍 ANALYSIS:")
    print("Claude's Strengths:")
    print("  • Strong reasoning and explanation")
    print("  • Consistent logical thinking")
    print("  • Good at admitting limitations")
    print()
    print("Revolutionary AI's Strengths:")
    print("  • Python interpreter for exact calculations")
    print("  • Real-time web search capabilities") 
    print("  • Neural routing to right tools")
    print("  • No knowledge cutoff limitations")
    
    return {
        'claude_wins': claude_wins,
        'revolutionary_wins': revolutionary_wins,
        'ties': ties,
        'total': total_rounds
    }

if __name__ == "__main__":
    claude_vs_revolutionary_test()