# NeuroTiny

A no-BPE neural language model with VQ-VAE tokenization, tiny experts, routing, and speculative decoding.

## Overview

NeuroTiny implements a complete neural language model pipeline with:

- **NeuroTokenizer**: VQ-VAE-based neural tokenizer (no BPE)
- **Tiny Experts**: Reason-mini (planning) and Struct-mini (JSON generation)
- **Router**: Fast/slow path routing with complexity classification  
- **Verification**: Schema validation and minimal repair
- **Speculative Decoding**: Draft prediction for faster inference
- **Tool Integration**: Web scraping with provenance tracking

## Quick Start

```bash
# Setup environment
make setup

# Train models (quick mode)
make train

# Run demo
make demo

# Full overnight training
make overnight
```

## Architecture

```
Query → NeuroTokenizer → Router → [Fast/Slow Path] → Experts → Verifier → JSON
                                      ↓
                              Tools + Evidence
```

### Components

- `neurotok/`: VQ-VAE tokenizer with encode/decode
- `experts/`: Reason-mini and Struct-mini models
- `router/`: Path routing and task classification
- `tools/`: Web adapter and evidence extraction
- `verify/`: Schema validation and JSON repair
- `constraints/`: JSON grammar for constrained generation
- `runtime/`: Main engine and speculative decoding

## Training

Train the complete system:

```bash
# Quick training (30 minutes)
make train

# Full overnight training on H100
make overnight
```

## Usage

```python
from runtime.engine import RuntimeEngine

engine = RuntimeEngine()

result = engine.run(
    query="Get iPhone 14 Pro price",
    wants_json=True,
    schema_id="product_v1"
)

print(result['json'])
```

## Benchmarks

Run the benchmark suite:

```bash
make benchmark
```

Expected performance:
- Compression: 2-3x over raw bytes
- Round-trip fidelity: >99.5%
- JSON validation: >95% pass rate
- Average latency: <1s per query

## File Structure

```
neurotok/           # VQ-VAE tokenizer
experts/           # Mini expert models
router/            # Routing system
tools/             # Web tools and adapters
verify/            # Validation and repair
constraints/       # JSON grammar
runtime/           # Main engine
bench/             # Benchmarks
data/              # Training data and traces
tests/             # Unit tests
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## License

MIT