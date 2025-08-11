#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from runtime.engine import RuntimeEngine


def run_demo():
    print("=" * 60)
    print("NeuroTiny Demo - Running example tasks")
    print("=" * 60)
    
    engine = RuntimeEngine(checkpoint_dir="checkpoints")
    
    examples = [
        {
            "name": "Simple Product Query",
            "query": "Get iPhone 14 Pro price and availability",
            "wants_json": True,
            "schema_id": "product_v1"
        },
        {
            "name": "Blog Post Extraction",
            "query": "Extract blog post about AI trends with author and tags",
            "wants_json": True,
            "url": "https://example.com/blog/ai-trends",
            "selector": ".post-content",
            "schema_id": "post_v1"
        },
        {
            "name": "Event Information",
            "query": "Get details for tech conference on March 15 at Convention Center",
            "wants_json": True,
            "schema_id": "event_v1"
        },
        {
            "name": "Product with Web Fetch",
            "query": "Fetch product details from shop",
            "wants_json": True,
            "url": "https://shop.example.com/product/laptop",
            "selector": ".product-info",
            "schema_id": "product_v1"
        }
    ]
    
    results = []
    
    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Running: {example['name']}")
        print(f"Query: {example['query']}")
        
        start = time.time()
        
        result = engine.run(
            query=example['query'],
            wants_json=example.get('wants_json', True),
            url=example.get('url'),
            selector=example.get('selector'),
            schema_id=example.get('schema_id')
        )
        
        elapsed = time.time() - start
        
        print(f"Path: {result['path']}")
        print(f"Status: {'✓' if result['ok'] else '✗'}")
        print(f"Fixed: {result.get('fixed', False)}")
        print(f"Latency: {elapsed:.3f}s")
        
        if result.get('json'):
            print("Output JSON:")
            print(json.dumps(result['json'], indent=2))
        
        if result.get('notes'):
            print(f"Notes: {result['notes']}")
        
        results.append({
            'example': example['name'],
            'success': result['ok'],
            'latency': elapsed,
            'path': result['path'],
            'output': result.get('json')
        })
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['success'])
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print(f"Success Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"Average Latency: {avg_latency:.3f}s")
    
    fast_path_count = sum(1 for r in results if r['path'] == 'FAST_PATH')
    print(f"Fast Path Usage: {fast_path_count}/{len(results)} ({fast_path_count/len(results)*100:.1f}%)")
    
    with open('bench/demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to bench/demo_results.json")
    
    compression_test = "This is a test string to check compression ratio and fidelity."
    codes = engine.tokenizer.encode(compression_test)
    decoded = engine.tokenizer.decode(codes)
    compression_ratio = engine.tokenizer.compress_ratio(compression_test)
    fidelity = engine.tokenizer.fidelity(compression_test)
    
    print(f"\nTokenizer Test:")
    print(f"Original: {compression_test}")
    print(f"Decoded:  {decoded}")
    print(f"Compression: {compression_ratio:.2f}x")
    print(f"Fidelity: {fidelity*100:.1f}%")


if __name__ == "__main__":
    run_demo()