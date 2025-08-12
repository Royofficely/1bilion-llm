#!/usr/bin/env python3
"""
REST API SERVER - Pure LLM Decision System
Create HTTP endpoint to test with curl requests
"""

from flask import Flask, request, jsonify
from pure_llm_decision_system import PureLLMInference
import time
import traceback

app = Flask(__name__)

# Global LLM instance
llm = None

def initialize_llm():
    """Initialize the LLM system"""
    global llm
    try:
        print("🤖 Loading Pure LLM Decision System...")
        llm = PureLLMInference()
        print("✅ Pure LLM loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    return jsonify({
        "message": "🤖 Pure LLM Decision System API",
        "status": "running",
        "endpoints": {
            "POST /ask": "Ask a question to the LLM",
            "GET /health": "Check system health",
            "GET /examples": "Get example questions"
        },
        "usage": {
            "curl_example": "curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{\"question\": \"What is 10 times 20?\"}'"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    global llm
    return jsonify({
        "status": "healthy" if llm else "not_ready",
        "llm_loaded": llm is not None,
        "timestamp": time.time()
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """Main endpoint to ask questions"""
    global llm
    
    if not llm:
        return jsonify({
            "error": "LLM not loaded",
            "message": "System is not ready. Please restart the server."
        }), 500
    
    try:
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing question",
                "message": "Please provide a 'question' field in JSON body"
            }), 400
            
        question = data['question'].strip()
        if not question:
            return jsonify({
                "error": "Empty question",
                "message": "Question cannot be empty"
            }), 400
        
        # Process with LLM
        start_time = time.time()
        
        # Get LLM decision and response (we'll extract decision info)
        response = llm.process_query(question)
        
        processing_time = time.time() - start_time
        
        # For better API response, we'll also get the decision info
        # (This is a simplified version - in production you'd modify the LLM to return structured data)
        decision_info = get_decision_info(question)
        
        return jsonify({
            "question": question,
            "answer": response,
            "decision": decision_info,
            "processing_time": round(processing_time, 4),
            "timestamp": time.time(),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "message": str(e),
            "traceback": traceback.format_exc() if app.debug else None
        }), 500

@app.route('/examples', methods=['GET'])
def examples():
    """Get example questions to try"""
    return jsonify({
        "examples": {
            "math": [
                "What is 47 times 83?",
                "Find the 15th Fibonacci number",
                "What is 2^8?",
                "Solve: 3x + 7 = 22",
                "Is 97 a prime number?"
            ],
            "text_processing": [
                "Reverse the word 'hello'",
                "Count the letter 's' in 'Mississippi'",
                "What's the first letter of 'psychology'?",
                "Check if 'listen' and 'silent' are anagrams"
            ],
            "knowledge": [
                "What is DNA?",
                "Capital of Australia",
                "What causes earthquakes?",
                "Explain photosynthesis"
            ],
            "programming": [
                "Write Python code to find prime numbers",
                "Create a function to reverse a string",
                "Python code for bubble sort"
            ]
        },
        "curl_examples": [
            "curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{\"question\": \"What is 25 times 34?\"}'",
            "curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{\"question\": \"Reverse the word hello\"}'",
            "curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{\"question\": \"What is DNA?\"}'",
            "curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{\"question\": \"Capital of France\"}'"
        ]
    })

@app.route('/batch', methods=['POST'])
def batch_questions():
    """Process multiple questions at once"""
    global llm
    
    if not llm:
        return jsonify({
            "error": "LLM not loaded",
            "message": "System is not ready. Please restart the server."
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'questions' not in data:
            return jsonify({
                "error": "Missing questions",
                "message": "Please provide a 'questions' array in JSON body"
            }), 400
            
        questions = data['questions']
        if not isinstance(questions, list) or len(questions) == 0:
            return jsonify({
                "error": "Invalid questions",
                "message": "Questions must be a non-empty array"
            }), 400
            
        if len(questions) > 10:
            return jsonify({
                "error": "Too many questions",
                "message": "Maximum 10 questions per batch request"
            }), 400
        
        # Process all questions
        results = []
        total_start = time.time()
        
        for i, question in enumerate(questions):
            if not question or not question.strip():
                results.append({
                    "question": question,
                    "error": "Empty question",
                    "status": "failed"
                })
                continue
                
            try:
                start_time = time.time()
                response = llm.process_query(question.strip())
                processing_time = time.time() - start_time
                
                results.append({
                    "question": question.strip(),
                    "answer": response,
                    "processing_time": round(processing_time, 4),
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "question": question.strip(),
                    "error": str(e),
                    "status": "failed"
                })
        
        total_time = time.time() - total_start
        
        return jsonify({
            "results": results,
            "total_questions": len(questions),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "total_processing_time": round(total_time, 4),
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "error": "Batch processing failed",
            "message": str(e)
        }), 500

def get_decision_info(question):
    """Extract decision information from question (simplified)"""
    question_lower = question.lower()
    
    # Simple heuristics to show decision info (in production, extract from LLM)
    if any(word in question_lower for word in ['times', 'plus', 'minus', 'divide', 'calculate']):
        return {"problem_type": "arithmetic", "method": "direct_calculation"}
    elif any(word in question_lower for word in ['reverse', 'count', 'letter', 'first']):
        return {"problem_type": "text_processing", "method": "transformation"}
    elif any(word in question_lower for word in ['what is', 'explain', 'capital', 'causes']):
        return {"problem_type": "knowledge", "method": "factual_recall"}
    elif any(word in question_lower for word in ['python', 'code', 'function']):
        return {"problem_type": "programming", "method": "algorithm"}
    elif any(word in question_lower for word in ['fibonacci', 'sequence', 'prime']):
        return {"problem_type": "mathematical_reasoning", "method": "pattern_recognition"}
    else:
        return {"problem_type": "general", "method": "inference"}

if __name__ == '__main__':
    print("🚀 Starting Pure LLM Decision System API Server")
    print("=" * 60)
    
    # Initialize LLM
    if initialize_llm():
        print("\n🌐 Starting Flask server...")
        print("📍 Server will be available at: http://localhost:5000")
        print("\n📝 Test with curl:")
        print("curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{\"question\": \"What is 10 times 20?\"}'")
        print("\n🔍 Other endpoints:")
        print("• GET  http://localhost:5000/         - Home page")
        print("• GET  http://localhost:5000/health   - Health check")  
        print("• GET  http://localhost:5000/examples - Example questions")
        print("• POST http://localhost:5000/batch    - Multiple questions")
        print("\n" + "="*60)
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("❌ Failed to initialize LLM. Please run training first:")
        print("python3 pure_llm_decision_system.py")