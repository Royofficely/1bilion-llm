# 🚀 Pure LLM Decision System API - Curl Tests

## Server Status
✅ **Server is running at: http://localhost:5000**

## Quick Tests

### 1. Basic Health Check
```bash
curl -X GET http://localhost:5000/health
```

### 2. Math Questions
```bash
# Multiplication
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "What is 25 times 34?"}'

# Fibonacci
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "Find the 15th Fibonacci number"}'

# Powers
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "What is 2^8?"}'

# Algebra
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "Solve: 3x + 7 = 22"}'
```

### 3. Text Processing
```bash
# Word reversal
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "Reverse the word extraordinary"}'

# Letter counting
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "Count the letter s in Mississippi"}'

# First letter
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "What is the first letter of psychology?"}'
```

### 4. Knowledge Questions
```bash
# Science
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "What is DNA?"}'

# Geography  
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "Capital of Australia"}'

# Natural phenomena
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "What causes earthquakes?"}'
```

### 5. Programming Questions
```bash
# Code generation
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "Write Python code to find prime numbers"}'

# Function creation
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "Create a function to reverse a string"}'
```

### 6. Batch Processing (Multiple Questions)
```bash
curl -X POST http://localhost:5000/batch -H 'Content-Type: application/json' -d '{
  "questions": [
    "What is 10 times 20?",
    "Reverse the word hello",
    "What is DNA?",
    "Capital of France"
  ]
}'
```

### 7. Get Examples
```bash
curl -X GET http://localhost:5000/examples
```

## Expected Response Format

Each question returns JSON with:
```json
{
  "question": "What is 25 times 34?",
  "answer": "25 × 34 = 850.0",
  "decision": {
    "problem_type": "arithmetic",
    "method": "direct_calculation"
  },
  "processing_time": 0.0063,
  "timestamp": 1755005470.8207932,
  "status": "success"
}
```

## Key Features

- **🧠 LLM Decisions**: See how the LLM categorizes each problem
- **⚡ Fast Processing**: ~0.003-0.006 seconds per query
- **🎯 Structured Output**: JSON responses with decision info
- **📊 Batch Support**: Process multiple questions at once
- **🔍 Health Monitoring**: Check system status

## Problem Types the LLM Recognizes

- `arithmetic` - Basic math operations
- `text_processing` - String manipulation
- `knowledge` - Factual questions  
- `programming` - Code generation
- `mathematical_reasoning` - Complex math patterns
- `general` - Other reasoning tasks

## Solution Methods the LLM Chooses

- `direct_calculation` - Compute directly
- `transformation` - Transform input (like reversing)
- `factual_recall` - Look up knowledge
- `algorithm` - Use programming algorithm
- `pattern_recognition` - Find patterns
- `inference` - Logical reasoning

## Testing Tips

1. **Start Simple**: Try basic math first
2. **Check Decision Quality**: See if problem_type makes sense
3. **Time Performance**: Processing should be under 0.01s
4. **Try Edge Cases**: Empty questions, very long questions
5. **Batch Test**: Test multiple questions for efficiency

## Stop the Server
Press `Ctrl+C` in the terminal where the server is running, or:
```bash
pkill -f api_server.py
```