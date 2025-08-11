#!/usr/bin/env python3
"""
Quick verification script to check NeuroTiny setup without heavy dependencies.
"""

import os
import sys
from pathlib import Path

def check_file_structure():
    """Check if all required files exist."""
    print("📁 Checking file structure...")
    
    required_files = [
        'neurotok/vqvae.py',
        'neurotok/train_neurotok.py',
        'neurotok/api.py',
        'experts/reason_mini.py',
        'experts/struct_mini.py',
        'router/router.py',
        'router/classifier.py',
        'tools/web_adapter.py',
        'tools/evidence.py',
        'verify/schemas/product_v1.json',
        'verify/schemas/post_v1.json',
        'verify/schemas/event_v1.json',
        'verify/schema_check.py',
        'verify/repair.py',
        'constraints/json_grammar.py',
        'runtime/engine.py',
        'runtime/speculative.py',
        'runtime/formats.md',
        'bench/demo.py',
        'bench/run.py',
        'bench/tasks.md',
        'data/text_small.txt',
        'data/traces/all_traces.jsonl',
        'tests/test_basic.py',
        'Makefile',
        'README.md',
        'requirements.txt'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return False
    
    print("✅ All required files present!")
    return True

def check_json_schemas():
    """Test JSON schema loading."""
    print("\n🔍 Testing JSON schemas...")
    
    try:
        import json
        
        schemas = ['product_v1', 'post_v1', 'event_v1']
        for schema_name in schemas:
            path = f'verify/schemas/{schema_name}.json'
            with open(path, 'r') as f:
                schema = json.load(f)
            print(f"  ✅ {schema_name}: {len(schema.get('properties', {}))} fields")
        
        return True
    except Exception as e:
        print(f"  ❌ Schema error: {e}")
        return False

def check_traces():
    """Check trace files."""
    print("\n📊 Checking traces...")
    
    try:
        import json
        
        trace_files = list(Path('data/traces').glob('*.jsonl'))
        total_traces = 0
        
        for trace_file in trace_files:
            count = 0
            with open(trace_file, 'r') as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
                        count += 1
            total_traces += count
            print(f"  ✅ {trace_file.name}: {count} traces")
        
        print(f"  📈 Total traces: {total_traces}")
        return True
    except Exception as e:
        print(f"  ❌ Trace error: {e}")
        return False

def test_basic_imports():
    """Test basic module structure (without heavy dependencies)."""
    print("\n🐍 Testing basic imports...")
    
    try:
        # Test schema validation without torch
        sys.path.insert(0, '.')
        
        # Basic JSON operations
        import json
        from pathlib import Path
        
        # Test router logic (doesn't need torch)
        from router.router import Router
        router = Router()
        decision = router.route({
            'query': 'test query',
            'wants_json': True,
            'url': None
        })
        print(f"  ✅ Router test: {decision.path}")
        
        # Test schema validator
        from verify.schema_check import SchemaValidator
        validator = SchemaValidator()
        is_valid, errors = validator.validate_json({
            "name": "Test",
            "price": 99,
            "currency": "USD", 
            "in_stock": True,
            "url": "https://test.com"
        }, 'product_v1')
        print(f"  ✅ Schema validation: {is_valid}")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def main():
    print("🚀 NeuroTiny Setup Verification")
    print("=" * 40)
    
    checks = [
        check_file_structure(),
        check_json_schemas(),
        check_traces(),
        test_basic_imports()
    ]
    
    success_count = sum(checks)
    total_checks = len(checks)
    
    print(f"\n📋 Summary: {success_count}/{total_checks} checks passed")
    
    if success_count == total_checks:
        print("🎉 NeuroTiny setup verification PASSED!")
        print("\nNext steps:")
        print("  1. Install dependencies: make install")
        print("  2. Run training: make train")
        print("  3. Run demo: make demo")
        print("  4. For full training: make overnight")
    else:
        print("❌ Some checks failed. Please review the issues above.")
    
    return success_count == total_checks

if __name__ == "__main__":
    main()