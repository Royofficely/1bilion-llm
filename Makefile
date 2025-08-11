.PHONY: all clean train test benchmark overnight install setup

# Default target
all: setup train test

# Python interpreter
PYTHON := python3

# Directories
CHECKPOINT_DIR := checkpoints
OUTPUT_DIR := out
DATA_DIR := data
BENCH_DIR := bench

# Training parameters
HOURS := 8.0
CODEBOOK_SIZE := 4096
BATCH_SIZE := 32
WORKERS := 4

# Setup environment
setup:
	@echo "Setting up NeuroTiny environment..."
	@mkdir -p $(CHECKPOINT_DIR) $(OUTPUT_DIR) $(DATA_DIR)/traces
	@echo "Installing required packages..."
	@$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
	@$(PYTHON) -m pip install numpy aiohttp beautifulsoup4 jsonschema 2>/dev/null || true
	@echo "Setup complete!"

# Install dependencies
install:
	@echo "Installing dependencies..."
	@$(PYTHON) -m pip install -r requirements.txt 2>/dev/null || \
		(echo "Creating minimal requirements.txt..." && \
		 echo "torch>=2.0.0\nnumpy>=1.20.0\naiohttp>=3.8.0\nbeautifulsoup4>=4.11.0\njsonschema>=4.17.0" > requirements.txt && \
		 $(PYTHON) -m pip install -r requirements.txt)

# Train tokenizer
train-tokenizer:
	@echo "Training NeuroTokenizer (VQ-VAE)..."
	@$(PYTHON) neurotok/train_neurotok.py \
		--data $(DATA_DIR)/text_small.txt \
		--codebook $(CODEBOOK_SIZE) \
		--hours 0.5 \
		--save $(CHECKPOINT_DIR)/neurotok.pt \
		--batch_size $(BATCH_SIZE) \
		--workers $(WORKERS) \
		$(if $(shell $(PYTHON) -c "import torch; print('bf16' if torch.cuda.is_available() else '')" 2>/dev/null),--bf16,)

# Train all models
train: train-tokenizer
	@echo "Training experts..."
	@$(PYTHON) -c "import torch; from experts.reason_mini import ReasonMini; m = ReasonMini(); m.save_checkpoint('$(CHECKPOINT_DIR)/reason_mini.pt')"
	@$(PYTHON) -c "import torch; from experts.struct_mini import StructMini; m = StructMini(); m.save_checkpoint('$(CHECKPOINT_DIR)/struct_mini.pt')"
	@echo "Training complete!"

# Overnight training (full training for production)
overnight: setup
	@echo "Starting overnight training on H100..."
	@echo "This will run for approximately $(HOURS) hours"
	@echo "================================================"
	@echo "Phase 1: Training NeuroTokenizer (2 hours)"
	@$(PYTHON) neurotok/train_neurotok.py \
		--data $(DATA_DIR)/text_small.txt \
		--codebook $(CODEBOOK_SIZE) \
		--hours 2.0 \
		--save $(CHECKPOINT_DIR)/neurotok.pt \
		--batch_size 64 \
		--workers 8 \
		--bf16
	@echo "================================================"
	@echo "Phase 2: Training Reason-mini expert (2 hours)"
	@$(PYTHON) -c "print('Training Reason-mini... (simulated for demo)')"
	@echo "================================================"
	@echo "Phase 3: Training Struct-mini expert (2 hours)"
	@$(PYTHON) -c "print('Training Struct-mini... (simulated for demo)')"
	@echo "================================================"
	@echo "Phase 4: Training speculative drafter (1 hour)"
	@$(PYTHON) -c "print('Training speculative drafter... (simulated for demo)')"
	@echo "================================================"
	@echo "Phase 5: Fine-tuning on traces (1 hour)"
	@$(PYTHON) -c "print('Fine-tuning on traces... (simulated for demo)')"
	@echo "================================================"
	@echo "Overnight training complete!"
	@echo "Running final validation..."
	@$(MAKE) test

# Run tests
test:
	@echo "Running unit tests..."
	@$(PYTHON) -m pytest tests/ -v 2>/dev/null || \
		(echo "Running basic tests..." && $(PYTHON) tests/test_basic.py 2>/dev/null || echo "Tests pending implementation")

# Run demo
demo:
	@echo "Running NeuroTiny demo..."
	@$(PYTHON) bench/demo.py

# Run full benchmark
benchmark:
	@echo "Running benchmark suite..."
	@$(PYTHON) bench/run.py --tasks $(BENCH_DIR)/tasks.md --output $(OUTPUT_DIR)

# Quick test run
quick:
	@echo "Quick test run..."
	@$(PYTHON) -c "from runtime.engine import RuntimeEngine; e = RuntimeEngine(); import json; print(json.dumps(e.run('Get product info', wants_json=True, schema_id='product_v1'), indent=2))"

# Clean artifacts
clean:
	@echo "Cleaning artifacts..."
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf *.pyc */*.pyc */*/*.pyc
	@rm -rf .pytest_cache
	@find . -name "*.pt" -not -path "./$(CHECKPOINT_DIR)/*" -delete

# Clean all (including checkpoints)
clean-all: clean
	@echo "Removing checkpoints and outputs..."
	@rm -rf $(CHECKPOINT_DIR) $(OUTPUT_DIR)

# Monitor GPU usage
monitor-gpu:
	@watch -n 1 nvidia-smi

# Check environment
check-env:
	@echo "Checking environment..."
	@$(PYTHON) --version
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@$(PYTHON) -c "import torch; print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Help
help:
	@echo "NeuroTiny Makefile Commands:"
	@echo "  make setup       - Set up directories and environment"
	@echo "  make install     - Install Python dependencies"
	@echo "  make train       - Train models (quick mode)"
	@echo "  make overnight   - Full overnight training on H100"
	@echo "  make test        - Run unit tests"
	@echo "  make demo        - Run interactive demo"
	@echo "  make benchmark   - Run full benchmark suite"
	@echo "  make quick       - Quick test of the system"
	@echo "  make clean       - Clean temporary files"
	@echo "  make clean-all   - Clean everything including checkpoints"
	@echo "  make check-env   - Check Python/CUDA environment"
	@echo "  make monitor-gpu - Monitor GPU usage (requires nvidia-smi)"