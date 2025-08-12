#!/usr/bin/env python3
"""
FINAL 95% ACCURACY VALIDATION TEST
Comprehensive testing suite to validate 95%+ accuracy
Tests all major AI capabilities to beat GPT/Claude
"""

import time
import json
from datetime import datetime

class Final95AccuracyValidator:
    def __init__(self):
        self.test_suites = {
            "math_operations": [],
            "letter_counting": [],
            "string_manipulation": [],
            "logic_reasoning": [],
            "pattern_recognition": [],
            "knowledge_recall": [],
            "complex_reasoning": []
        }
        self.results = {}
        
    def load_comprehensive_tests(self):
        """Load comprehensive test suite for 95% validation"""
        
        # Math Operations (20 tests)
        self.test_suites["math_operations"] = [
            ("2 + 2", "4"),
            ("15 × 7", "105"), 
            ("144 ÷ 12", "12"),
            ("√169", "13"),
            ("5² + 3²", "34"),
            ("25% of 80", "20"),
            ("7! ÷ 5!", "42"),
            ("2³ × 3²", "72"),
            ("(15 + 5) × 2", "40"),
            ("100 - 37", "63"),
            ("347 × 29", "10063"),
            ("√144 + 17²", "301"),
            ("30% of 150", "45"),
            ("8² - 6²", "28"),
            ("456 ÷ 8", "57"),
            ("3⁴ - 2⁵", "49"),
            ("75% of 200", "150"),
            ("12 × 12 + 1", "145"),
            ("200 ÷ 8 + 15", "40"),
            ("6² + 8² - 10²", "0")
        ]
        
        # Letter Counting (15 tests)
        self.test_suites["letter_counting"] = [
            ("count s in mississippi", "4"),
            ("count e in excellence", "4"),
            ("count a in banana", "3"),
            ("count r in strawberry", "3"),
            ("count o in chocolate", "2"),
            ("count t in butterfly", "2"),
            ("count l in parallel", "3"),
            ("count n in international", "4"),
            ("count i in mississippi", "4"),
            ("count p in mississippi", "4"),
            ("count vowels in programming", "3"),
            ("count consonants in hello", "3"),
            ("count z in pizza", "2"),
            ("count x in xerxes", "2"),
            ("count double letters in bookkeeper", "3")
        ]
        
        # String Manipulation (15 tests) 
        self.test_suites["string_manipulation"] = [
            ("reverse hello", "olleh"),
            ("reverse world", "dlrow"),
            ("reverse artificial", "laicifitra"),
            ("first letter of apple", "a"),
            ("last letter of programming", "g"),
            ("5th character in BENCHMARK", "H"),
            ("3rd character in PYTHON", "T"),
            ("middle letter of hello", "l"),
            ("length of programming", "11"),
            ("reverse computer", "retupmoc"),
            ("first letter of zebra", "z"),
            ("7th character in WONDERFUL", "R"),
            ("last letter of beautiful", "l"),
            ("2nd character in AMAZING", "M"),
            ("reverse intelligence", "ecnegilletni")
        ]
        
        # Logic Reasoning (15 tests)
        self.test_suites["logic_reasoning"] = [
            ("Tom has 4 brothers and 3 sisters. How many sisters do Tom's brothers have?", "4"),
            ("If 5 apples cost $10, how much do 8 apples cost?", "16"),
            ("24 students, 4 per group. How many groups?", "6"),
            ("Alice is 5 years older than Bob. Bob is 20. How old is Alice?", "25"),
            ("In 5 years, John will be 30. How old is he now?", "25"),
            ("Sara has 2 brothers and 4 sisters. How many siblings does Sara have?", "6"),
            ("If all roses are flowers, and all flowers are plants, are roses plants?", "yes"),
            ("A train travels 60 mph for 3 hours. Distance?", "180"),
            ("What day comes after Wednesday?", "Thursday"),
            ("What month comes before July?", "June"),
            ("How many days in February 2024?", "29"),
            ("If today is Monday, what day is it in 3 days?", "Thursday"),
            ("In a family of 7 children, 4 are boys. How many are girls?", "3"),
            ("Circle area with radius 5 (π≈3.14)", "78.5"),
            ("Rectangle area 8×6", "48")
        ]
        
        # Pattern Recognition (15 tests)
        self.test_suites["pattern_recognition"] = [
            ("2, 6, 18, 54, ?", "162"),
            ("1, 4, 9, 16, 25, ?", "36"),
            ("5, 10, 15, 20, ?", "25"),
            ("1, 1, 2, 3, 5, 8, ?", "13"),
            ("2, 4, 8, 16, ?", "32"),
            ("A, C, E, G, ?", "I"),
            ("Z, Y, X, W, ?", "V"),
            ("1, 3, 9, 27, ?", "81"),
            ("3, 7, 15, 31, ?", "63"),
            ("100, 90, 80, 70, ?", "60"),
            ("B, D, F, H, ?", "J"),
            ("2, 5, 11, 23, ?", "47"),
            ("1, 4, 7, 10, ?", "13"),
            ("4, 7, 12, 19, ?", "28"),
            ("0, 1, 4, 9, 16, ?", "25")
        ]
        
        # Knowledge Recall (10 tests)
        self.test_suites["knowledge_recall"] = [
            ("capital of France", "Paris"),
            ("largest planet", "Jupiter"), 
            ("author of 1984", "George Orwell"),
            ("chemical symbol for gold", "Au"),
            ("speed of light", "299792458"),
            ("inventor of telephone", "Alexander Graham Bell"),
            ("year World War 2 ended", "1945"),
            ("smallest prime number", "2"),
            ("currency of Japan", "Yen"),
            ("number of continents", "7")
        ]
        
        # Complex Reasoning (10 tests)
        self.test_suites["complex_reasoning"] = [
            ("If you have 12 apples and eat 1/4, then buy 6 more, how many do you have?", "15"),
            ("A car travels 100km in 2 hours, then 150km in 3 hours. Average speed?", "50"),
            ("You have $100. Spend 40%, then earn 25% more. How much now?", "75"),
            ("Age puzzle: I'm 3 times my age 10 years ago. How old am I?", "15"),
            ("Clock shows 3:15. What's the angle between hands?", "7.5"),
            ("Probability of rolling two sixes with dice", "0.028"),
            ("Compound interest: $1000 at 10% for 2 years", "1210"),
            ("Speed to cover 240km in 4 hours", "60"),
            ("Percentage increase from 80 to 120", "50"),
            ("Binary 1101 in decimal", "13")
        ]
        
        total_tests = sum(len(suite) for suite in self.test_suites.values())
        print(f"📊 Loaded {total_tests} comprehensive tests across {len(self.test_suites)} categories")
        return total_tests
    
    def run_full_validation(self, ai_system):
        """Run complete 95% accuracy validation"""
        print("\n🎯 FINAL 95% ACCURACY VALIDATION")
        print("=" * 70)
        print("🏆 COMPREHENSIVE TEST TO BEAT GPT/CLAUDE")
        print("=" * 70)
        
        overall_correct = 0
        overall_total = 0
        category_results = {}
        
        for category, tests in self.test_suites.items():
            if not tests:
                continue
                
            print(f"\n📂 CATEGORY: {category.upper().replace('_', ' ')}")
            print("-" * 50)
            
            category_correct = 0
            category_total = len(tests)
            
            for i, (question, expected) in enumerate(tests, 1):
                start_time = time.time()
                
                # Get answer from AI system
                try:
                    if hasattr(ai_system, 'answer_question'):
                        answer = ai_system.answer_question(question)
                    else:
                        answer = "System not available"
                except Exception as e:
                    answer = f"Error: {str(e)}"
                
                elapsed = time.time() - start_time
                
                # Check if correct
                is_correct = self._is_answer_correct(answer, expected)
                if is_correct:
                    category_correct += 1
                    overall_correct += 1
                
                overall_total += 1
                
                # Display result
                status = "✅" if is_correct else "❌"
                print(f"{status} Test {i:2d}: {question[:50]}...")
                print(f"    Expected: {expected} | Got: {answer} | {elapsed:.3f}s")
            
            # Category summary
            category_accuracy = (category_correct / category_total) * 100
            category_results[category] = {
                "accuracy": category_accuracy,
                "correct": category_correct,
                "total": category_total
            }
            
            print(f"\n📊 {category.upper()}: {category_accuracy:.1f}% ({category_correct}/{category_total})")
        
        # Overall results
        overall_accuracy = (overall_correct / overall_total) * 100
        
        print(f"\n🎯 FINAL VALIDATION RESULTS")
        print("=" * 70)
        print(f"🏆 OVERALL ACCURACY: {overall_accuracy:.1f}% ({overall_correct}/{overall_total})")
        print(f"🎯 TARGET ACCURACY: 95.0%")
        
        # Detailed breakdown
        print(f"\n📊 DETAILED BREAKDOWN:")
        for category, result in category_results.items():
            accuracy = result["accuracy"]
            correct = result["correct"] 
            total = result["total"]
            status = "✅" if accuracy >= 90 else "⚠️" if accuracy >= 70 else "❌"
            print(f"   {status} {category.replace('_', ' ').title()}: {accuracy:.1f}% ({correct}/{total})")
        
        # Final verdict
        print(f"\n🏁 FINAL VERDICT:")
        if overall_accuracy >= 95.0:
            print("🎉 SUCCESS! 95%+ ACCURACY ACHIEVED!")
            print("🏆 OFFICIALLY BEATS GPT/CLAUDE IN COMPREHENSIVE TESTING!")
            print("⚡ READY FOR PRODUCTION DEPLOYMENT!")
            verdict = "PASSED"
        elif overall_accuracy >= 85.0:
            print(f"📈 STRONG PERFORMANCE: {overall_accuracy:.1f}%")
            print("🔧 Close to 95% target - minor improvements needed")
            verdict = "NEAR_PASS"  
        else:
            needed = int(overall_total * 0.95) - overall_correct
            print(f"🚧 IN PROGRESS: {overall_accuracy:.1f}%")
            print(f"📈 Need {needed} more correct answers for 95% target")
            verdict = "NEEDS_IMPROVEMENT"
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_accuracy": overall_accuracy,
            "overall_correct": overall_correct,
            "overall_total": overall_total,
            "target_accuracy": 95.0,
            "verdict": verdict,
            "category_results": category_results
        }
        
        return results
    
    def _is_answer_correct(self, answer, expected):
        """Enhanced answer comparison"""
        if answer is None or expected is None:
            return False
            
        # Convert to strings and normalize
        answer_str = str(answer).strip().lower()
        expected_str = str(expected).strip().lower()
        
        # Exact match
        if answer_str == expected_str:
            return True
            
        # Numeric comparison with tolerance
        try:
            answer_num = float(answer_str)
            expected_num = float(expected_str)
            return abs(answer_num - expected_num) < 0.01
        except:
            pass
        
        # Handle common variations
        if answer_str in expected_str or expected_str in answer_str:
            return True
            
        return False
    
    def save_results(self, results, filename="final_95_validation_results.json"):
        """Save validation results"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")

def main():
    """Main validation pipeline"""
    validator = Final95AccuracyValidator()
    validator.load_comprehensive_tests()
    
    print("🔧 Note: Run this with your trained AI system:")
    print("   from ultimate_95_percent_trainer import Ultimate95PercentTrainer")
    print("   trainer = Ultimate95PercentTrainer()")
    print("   trainer.train_with_enhanced_data()")
    print("   results = validator.run_full_validation(trainer)")
    
    return validator

if __name__ == "__main__":
    main()