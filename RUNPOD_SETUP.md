# 🚀 Runpod Setup Instructions

## Quick Setup on Runpod

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/pure-llm-decision-system.git
cd pure-llm-decision-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Enhanced System
```bash
# Generate massive training dataset and train
python3 pure_llm_decision_system.py
```

### 4. Test the System
```bash
# Start API server
python3 api_server.py &

# Test with curl
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "What is 47 times 83?"}'
```

## Key Files
- pure_llm_decision_system.py - Main LLM training
- api_server.py - REST API server  
- enhanced_training_system.py - Enhanced training
- pure_llm_comprehensive_test.py - Test suite
