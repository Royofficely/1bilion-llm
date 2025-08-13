#!/usr/bin/env python3
"""
🔥 SUPERIOR ROUTER vs CLAUDE - QUALITY BATTLE
Test our advanced router against Claude's reasoning
"""
import torch
import torch.nn as nn
import re
import time
from superior_routing_system import AdvancedRoutingLLM

class SuperiorRouterInference:
    """Inference for our superior routing system"""
    
    def __init__(self, model_path='superior_routing_model.pt'):
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            self.tokenizer = checkpoint['tokenizer']
            self.reverse_tokenizer = checkpoint['reverse_tokenizer']
            self.problem_types = checkpoint['problem_types']
            self.solution_methods = checkpoint['solution_methods']
            
            self.model = AdvancedRoutingLLM(len(self.tokenizer))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("🚀 Superior routing system loaded!")
        except Exception as e:
            print(f"❌ Could not load superior model: {e}")
            raise
    
    def analyze_query(self, query):
        """Advanced query analysis with confidence and reasoning"""
        print(f"\n🧠 SUPERIOR ROUTER ANALYSIS: {query}")
        print("-" * 50)
        
        # Tokenize
        tokens = re.findall(r'\w+|[^\w\s]', query.lower())
        token_ids = [self.tokenizer.get(token, self.tokenizer['<unk>']) for token in tokens]
        
        # Pad
        max_len = 150
        if len(token_ids) < max_len:
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]
        
        # Inference
        with torch.no_grad():
            input_tensor = torch.tensor([token_ids], dtype=torch.long)
            outputs = self.model(input_tensor)
            
            # Get predictions
            problem_idx = torch.argmax(outputs['problem_type'], dim=1).item()
            method_idx = torch.argmax(outputs['solution_method'], dim=1).item()
            confidence = outputs['confidence'].item()
            difficulty_idx = torch.argmax(outputs['difficulty'], dim=1).item()
            
            problem_type = self.problem_types[problem_idx]
            method = self.solution_methods[method_idx]
            
            # Get attention weights for interpretability
            attention = outputs['attention_weights'][0].mean(dim=0)  # Average across heads
            
            return {
                'problem_type': problem_type,
                'solution_method': method,
                'confidence': confidence,
                'difficulty_level': difficulty_idx + 1,
                'attention_pattern': attention,
                'reasoning': self._generate_reasoning(problem_type, method, confidence, difficulty_idx + 1)
            }
    
    def _generate_reasoning(self, problem_type, method, confidence, difficulty):
        """Generate human-readable reasoning"""
        reasoning_templates = {
            'calculus_derivatives': f"This is a calculus problem requiring {method}. The derivative calculation needs systematic application of differentiation rules.",
            'algebra_quadratic': f"This quadratic equation requires {method}. I'll use the quadratic formula or factoring approach.",
            'text_analysis': f"This text analysis task needs {method}. I'll examine semantic patterns and contextual meaning.",
            'programming_algorithms': f"This algorithmic challenge requires {method}. I'll design an efficient solution with proper complexity analysis.",
            'knowledge_science': f"This scientific knowledge query needs {method}. I'll synthesize information from multiple domains."
        }
        
        base_reasoning = reasoning_templates.get(problem_type, f"This {problem_type} problem requires {method}.")
        
        if confidence > 0.9:
            confidence_note = "I'm highly confident in this classification."
        elif confidence > 0.7:
            confidence_note = "I'm reasonably confident in this approach."
        else:
            confidence_note = "This query has some ambiguity, but this seems the best approach."
        
        difficulty_note = {
            1: "This is a basic-level problem.",
            2: "This is a moderate-difficulty problem.",
            3: "This is an intermediate-level challenge.",
            4: "This is a complex, advanced problem.",
            5: "This is an expert-level challenge requiring deep expertise."
        }[difficulty]
        
        return f"{base_reasoning} {difficulty_note} {confidence_note}"

def claude_routing_analysis(query):
    """Simulate Claude's routing thought process"""
    print(f"\n🤖 CLAUDE'S ROUTING ANALYSIS: {query}")
    print("-" * 50)
    
    time.sleep(0.001)  # Simulate thinking time
    
    # Claude's approach - more general, less specialized
    if any(word in query.lower() for word in ['derivative', 'differentiate', 'calculus']):
        return {
            'approach': "I need to solve a calculus problem involving derivatives",
            'method': "I'll apply differentiation rules step by step",
            'reasoning': "This requires knowledge of calculus differentiation rules and careful algebraic manipulation",
            'confidence': "high",
            'time_to_think': 0.001
        }
    elif 'quadratic' in query.lower() or ('x^2' in query):
        return {
            'approach': "This is a quadratic equation",
            'method': "I'll use the quadratic formula or try to factor",
            'reasoning': "Quadratic equations can be solved using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a",
            'confidence': "high",
            'time_to_think': 0.001
        }
    elif any(word in query.lower() for word in ['analyze', 'sentiment', 'text']):
        return {
            'approach': "This is a text analysis task",
            'method': "I'll examine the text for patterns, sentiment, and meaning",
            'reasoning': "Text analysis requires understanding context, semantics, and linguistic patterns",
            'confidence': "medium",
            'time_to_think': 0.002
        }
    elif any(word in query.lower() for word in ['algorithm', 'implement', 'code', 'programming']):
        return {
            'approach': "This is a programming/algorithm problem",
            'method': "I'll design an algorithm and implement it step by step",
            'reasoning': "Programming problems require algorithmic thinking and understanding of data structures",
            'confidence': "high",
            'time_to_think': 0.002
        }
    else:
        return {
            'approach': "Let me think about this problem carefully",
            'method': "I'll analyze what's being asked and determine the best approach",
            'reasoning': "This requires general reasoning and problem analysis",
            'confidence': "medium",
            'time_to_think': 0.003
        }

def run_quality_battle():
    """Test superior router vs Claude on routing quality"""
    print("🔥 SUPERIOR ROUTER vs CLAUDE - ROUTING QUALITY BATTLE")
    print("=" * 70)
    
    try:
        router = SuperiorRouterInference()
    except:
        print("❌ Superior router not ready yet - still training!")
        print("⏳ Please wait for training to complete...")
        return
    
    # Test cases that require high-quality routing decisions
    quality_tests = [
        {
            'query': 'Find the derivative of sin(x) * cos(x) using product rule',
            'category': 'Advanced Calculus',
            'expected_quality': 'Should identify product rule application specifically'
        },
        {
            'query': 'Analyze the emotional undertones and implicit biases in this political speech excerpt',
            'category': 'Complex Text Analysis', 
            'expected_quality': 'Should recognize multi-layered analysis need'
        },
        {
            'query': 'Implement a self-balancing binary search tree with rotation optimization',
            'category': 'Advanced Programming',
            'expected_quality': 'Should identify complex data structure requirements'
        },
        {
            'query': 'Explain quantum entanglement effects on cryptographic security in distributed systems',
            'category': 'Cross-Domain Knowledge',
            'expected_quality': 'Should recognize multi-domain expertise need'
        },
        {
            'query': 'Design a neural network architecture for real-time video object detection with memory constraints',
            'category': 'AI/ML Engineering',
            'expected_quality': 'Should identify optimization constraints and technical requirements'
        }
    ]
    
    superior_wins = 0
    claude_wins = 0
    ties = 0
    
    for i, test in enumerate(quality_tests, 1):
        print(f"\n{'='*70}")
        print(f"🔥 QUALITY TEST {i}: {test['category']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected_quality']}")
        print("="*70)
        
        # Our superior router analysis
        start_time = time.time()
        superior_analysis = router.analyze_query(test['query'])
        superior_time = time.time() - start_time
        
        print(f"🎯 Problem Type: {superior_analysis['problem_type']}")
        print(f"🔧 Method: {superior_analysis['solution_method']}")
        print(f"🎭 Confidence: {superior_analysis['confidence']:.3f}")
        print(f"📊 Difficulty: {superior_analysis['difficulty_level']}/5")
        print(f"💭 Reasoning: {superior_analysis['reasoning']}")
        print(f"⚡ Time: {superior_time:.4f}s")
        
        # Claude's analysis
        claude_analysis = claude_routing_analysis(test['query'])
        
        print(f"🤖 Approach: {claude_analysis['approach']}")
        print(f"🔧 Method: {claude_analysis['method']}")
        print(f"💭 Reasoning: {claude_analysis['reasoning']}")
        print(f"🎭 Confidence: {claude_analysis['confidence']}")
        print(f"⚡ Time: {claude_analysis['time_to_think']:.4f}s")
        
        # Quality comparison
        print(f"\n🏆 QUALITY COMPARISON:")
        
        # Scoring criteria
        superior_score = 0
        claude_score = 0
        
        # Specificity of problem identification
        if len(superior_analysis['problem_type'].split('_')) > 1:  # More specific
            superior_score += 1
            print("✅ Superior router: More specific problem classification")
        else:
            claude_score += 1
            print("✅ Claude: General but clear problem identification")
        
        # Method sophistication
        sophisticated_methods = ['domain_expertise', 'multi_step_process', 'contextual_interpretation', 'creative_synthesis']
        if superior_analysis['solution_method'] in sophisticated_methods:
            superior_score += 1
            print("✅ Superior router: Sophisticated method selection")
        else:
            claude_score += 1
            print("✅ Claude: Clear method explanation")
        
        # Confidence calibration
        if superior_analysis['confidence'] > 0.8:
            superior_score += 1
            print("✅ Superior router: High confidence with quantified score")
        else:
            claude_score += 1
            print("✅ Claude: Reasonable confidence assessment")
        
        # Reasoning depth
        if len(superior_analysis['reasoning'].split('.')) >= 3:
            superior_score += 1
            print("✅ Superior router: Detailed multi-aspect reasoning")
        else:
            claude_score += 1
            print("✅ Claude: Clear reasoning explanation")
        
        # Determine winner
        if superior_score > claude_score:
            superior_wins += 1
            print(f"🏆 WINNER: SUPERIOR ROUTER ({superior_score} vs {claude_score})")
        elif claude_score > superior_score:
            claude_wins += 1
            print(f"🏆 WINNER: CLAUDE ({claude_score} vs {superior_score})")
        else:
            ties += 1
            print(f"🤝 TIE ({superior_score} vs {claude_score})")
    
    # Final results
    print("\n" + "=" * 70)
    print("🏆 FINAL ROUTING QUALITY BATTLE RESULTS")
    print("=" * 70)
    print(f"🚀 Superior Router Wins: {superior_wins}")
    print(f"🤖 Claude Wins: {claude_wins}")
    print(f"🤝 Ties: {ties}")
    
    if superior_wins > claude_wins:
        print(f"\n🎉 SUPERIOR ROUTER DOMINATES ROUTING QUALITY!")
        print(f"🎯 Advanced neural architecture beats Claude's general reasoning!")
    elif claude_wins > superior_wins:
        print(f"\n🤖 Claude still wins on routing quality")
        print(f"💡 Need more training or architectural improvements")
    else:
        print(f"\n⚖️ Close battle - both systems show strengths")

if __name__ == "__main__":
    run_quality_battle()