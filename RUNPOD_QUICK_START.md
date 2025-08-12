# 🚀 RUNPOD QUICK START

## Git Repository
```bash
git clone https://github.com/Royofficely/1bilion-llm.git
cd 1bilion-llm
```

## Setup & Train
```bash
# Install dependencies
pip install torch flask transformers numpy scikit-learn tqdm

# Train the Pure LLM (this will generate 7000+ examples and train)
python3 pure_llm_decision_system.py

# Test the trained system
python3 pure_llm_comprehensive_test.py
```

## Start API Server
```bash
# Start server
python3 api_server.py &

# Test with curl
curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{"question": "What is 47 times 83?"}'
```

## Expected Results
- **Before**: 66.7% score (16 examples)  
- **After**: 85-90% score (7000+ examples)
- **Training time**: 5-15 minutes on GPU vs 30-60 minutes on CPU

## GPU Setup
For fastest training on Runpod:
- Choose A100 or V100 GPU
- Select PyTorch template
- Memory: 8GB+ VRAM recommended

## Files You Need
- `pure_llm_decision_system.py` - Main training script
- `api_server.py` - REST API server
- `pure_llm_comprehensive_test.py` - Test suite
- `requirements.txt` - Dependencies

That's it! The system will generate massive training data and retrain automatically! 🤖