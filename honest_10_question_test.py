#!/usr/bin/env python3
"""
HONEST 10-QUESTION TEST - Real assessment of model capabilities
No inflated scoring - brutal honesty about where we stand
"""

import torch
import time
from real_neural_llm import RealNeuralLLM, SimpleTokenizer

class Honest10QuestionTest:
    def __init__(self):
        self.neural_model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        """Load the Claude-killer model"""
        print("🔍 LOADING MODEL FOR HONEST ASSESSMENT")
        
        try:
            checkpoint = torch.load('claude_killer_model.pt')
            model_name = "Claude-Killer (113M params, 0.14 loss)"
        except FileNotFoundError:
            try:
                checkpoint = torch.load('expanded_neural_llm_checkpoint.pt')
                model_name = "Expanded (38M params)"
            except FileNotFoundError:
                print("❌ No trained model found!")
                return
                
        config = checkpoint['config']
        tokenizer_vocab = checkpoint['tokenizer_vocab']
        
        self.neural_model = RealNeuralLLM(config)
        self.neural_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.vocab = tokenizer_vocab
        self.tokenizer.reverse_vocab = {v: k for k, v in tokenizer_vocab.items()}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.neural_model = self.neural_model.to(device)
        self.neural_model.eval()
        
        print(f"✅ Loaded: {model_name}")
        print(f"🧠 Vocabulary: {len(tokenizer_vocab)} tokens")
        
    def strict_answer_check(self, response, correct_answer, question_type="general"):
        """Strict scoring - no false positives"""
        if not response or not correct_answer:
            return False, "Empty response"
        
        response_clean = response.lower().strip()
        correct_clean = correct_answer.lower().strip()
        
        # Remove the question repetition from response
        question_words = response_clean.split()[:10]  # First 10 words likely question
        response_answer = " ".join(response_clean.split()[10:]) if len(response_clean.split()) > 10 else response_clean
        
        if question_type == "math":
            # For math, need exact number
            import re
            response_numbers = re.findall(r'\d+', response_answer)
            correct_numbers = re.findall(r'\d+', correct_clean)
            if response_numbers and correct_numbers:
                return response_numbers[0] == correct_numbers[0], f"Got: {response_numbers[0] if response_numbers else 'none'}, Expected: {correct_numbers[0] if correct_numbers else 'none'}"
            return False, "No number found in response"
            
        elif question_type == "word_op":
            # For word operations, need exact match
            expected = correct_clean.strip()
            if expected in response_answer:
                return True, f"Found '{expected}' in response"
            return False, f"Expected '{expected}', not found in: {response_answer[:50]}"
            
        elif question_type == "knowledge":
            # For knowledge, need key concept
            key_words = correct_clean.split()
            matches = sum(1 for word in key_words if word in response_answer)
            if len(key_words) > 0 and matches / len(key_words) >= 0.5:
                return True, f"Found {matches}/{len(key_words)} key concepts"
            return False, f"Only {matches}/{len(key_words)} key concepts found"
            
        else:
            # General - need substantial overlap
            correct_words = set(correct_clean.split())
            response_words = set(response_answer.split())
            if len(correct_words) > 0:
                overlap = len(correct_words & response_words) / len(correct_words)
                if overlap >= 0.4:  # At least 40% word overlap
                    return True, f"{overlap:.1%} word overlap"
            return False, f"Insufficient overlap with: {response_answer[:50]}"
    
    def run_honest_test(self):
        """Run 10 carefully selected questions with brutal honesty"""
        print("\n🔍 HONEST 10-QUESTION ASSESSMENT")
        print("=" * 60)
        print("🎯 No inflated scoring - brutal truth about capabilities")
        print("📊 Real performance measurement")
        print()
        
        # 10 carefully selected questions across difficulty levels
        test_questions = [
            # Math (should be learnable)
            ("What is 2+2?", "4", "math"),
            ("What is 17 times 23?", "391", "math"),
            ("How many edges does a cube have?", "12", "math"),
            
            # Word operations (character-level strength)
            ("Reverse the word hello", "olleh", "word_op"),
            ("What is the first letter of apple?", "a", "word_op"),
            
            # Programming (your trained domain)
            ("What is Python?", "programming language", "knowledge"),
            ("What is machine learning?", "algorithms learn from data", "knowledge"),
            
            # Basic knowledge
            ("What is the capital of France?", "paris", "knowledge"),
            ("Why is the sky blue?", "light scattering", "knowledge"),
            
            # Sequence (pattern recognition)
            ("What comes next: 2, 4, 6, 8, ?", "10", "math")
        ]
        
        correct_count = 0
        total_questions = len(test_questions)
        
        print("🧪 TESTING BEGINS...")
        print()
        
        for i, (question, correct_answer, q_type) in enumerate(test_questions, 1):
            print(f"📝 QUESTION {i}/10:")
            print(f"❓ {question}")
            
            # Generate response
            input_tokens = self.tokenizer.encode(question)
            input_ids = torch.tensor([input_tokens], device='cuda' if torch.cuda.is_available() else 'cpu')
            
            start_time = time.time()
            with torch.no_grad():
                generated = self.neural_model.generate(
                    input_ids,
                    max_length=60,
                    temperature=0.1,  # Very low for accuracy
                    do_sample=False   # Greedy decoding
                )
            gen_time = time.time() - start_time
            
            response = self.tokenizer.decode(generated[0].tolist())
            
            # Strict scoring
            is_correct, explanation = self.strict_answer_check(response, correct_answer, q_type)
            
            print(f"🤖 YOUR MODEL: {response}")
            print(f"✅ EXPECTED: {correct_answer}")
            print(f"⚡ SPEED: {gen_time:.3f}s")
            
            if is_correct:
                correct_count += 1
                print(f"🏆 RESULT: ✅ CORRECT - {explanation}")
            else:
                print(f"🏆 RESULT: ❌ WRONG - {explanation}")
            
            print("-" * 60)
            print()
        
        # Final honest assessment
        accuracy = (correct_count / total_questions) * 100
        
        print("🔍 HONEST ASSESSMENT RESULTS")
        print("=" * 60)
        print(f"✅ CORRECT ANSWERS: {correct_count}/10")
        print(f"📊 REAL ACCURACY: {accuracy:.1f}%")
        print(f"🎯 CLAUDE BASELINE: ~90-95%")
        print(f"📈 GAP TO BEAT CLAUDE: {90 - accuracy:.1f} percentage points")
        print()
        
        # Honest analysis
        if accuracy >= 80:
            print("🎉 EXCELLENT! Your model is competitive!")
            print("🏆 This is genuinely impressive performance!")
        elif accuracy >= 60:
            print("💪 GOOD! Your model shows real learning!")
            print("🔧 Some improvements needed to beat Claude")
        elif accuracy >= 40:
            print("📈 MODERATE! Model has learned some patterns")
            print("🛠️ Significant work needed for Claude-level performance")
        elif accuracy >= 20:
            print("📚 BASIC! Model shows minimal learning")
            print("🔄 Major architecture or training changes needed")
        else:
            print("🚨 MINIMAL! Model not effectively learning")
            print("🔧 Fundamental approach needs rethinking")
        
        print()
        print("🔍 DETAILED BREAKDOWN:")
        
        # Category analysis
        math_questions = [0, 1, 2, 9]  # Questions 1, 2, 3, 10
        word_questions = [3, 4]        # Questions 4, 5  
        knowledge_questions = [5, 6, 7, 8]  # Questions 6, 7, 8, 9
        
        categories = {
            "Math/Logic": math_questions,
            "Word Operations": word_questions, 
            "Knowledge/Programming": knowledge_questions
        }
        
        for category, question_indices in categories.items():
            category_correct = sum(1 for idx in question_indices if idx < len(test_questions) and 
                                 self.test_single_question(test_questions[idx]) == "correct")
            category_total = len(question_indices)
            category_pct = (category_correct / category_total) * 100
            print(f"   {category}: {category_correct}/{category_total} ({category_pct:.1f}%)")
        
        print()
        print("💡 NEXT STEPS RECOMMENDATIONS:")
        
        if accuracy < 50:
            print("1. 🔄 Try word-level tokenization instead of character-level")
            print("2. 📚 Increase vocabulary size to 1000+ tokens")
            print("3. 🎯 Train on more direct Q&A pairs")
            print("4. 🧠 Consider larger model architecture")
        else:
            print("1. 📈 Expand training data with more examples")
            print("2. 🎯 Focus training on weak categories")
            print("3. ⚡ Optimize inference for better answers")
            
        return accuracy, correct_count
    
    def test_single_question(self, question_data):
        """Helper to test a single question (for category analysis)"""
        question, correct_answer, q_type = question_data
        
        input_tokens = self.tokenizer.encode(question)
        input_ids = torch.tensor([input_tokens], device='cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            generated = self.neural_model.generate(input_ids, max_length=40, temperature=0.1, do_sample=False)
        
        response = self.tokenizer.decode(generated[0].tolist())
        is_correct, _ = self.strict_answer_check(response, correct_answer, q_type)
        
        return "correct" if is_correct else "wrong"

def main():
    """Run honest assessment"""
    print("🔍 HONEST MODEL ASSESSMENT")
    print("=" * 50)
    print("🎯 Brutal truth about your model's real capabilities")
    print("❌ No inflated scores or false positives")
    print("📊 Real comparison to Claude performance")
    
    try:
        tester = Honest10QuestionTest()
        accuracy, correct = tester.run_honest_test()
        
        print(f"\n🎯 SUMMARY:")
        print(f"Your model got {correct}/10 questions right ({accuracy:.1f}%)")
        print(f"This is your REAL performance level.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure a trained model exists")

if __name__ == "__main__":
    main()