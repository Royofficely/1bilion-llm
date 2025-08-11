#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from runtime.engine import RuntimeEngine


class BenchmarkRunner:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.engine = RuntimeEngine(checkpoint_dir=checkpoint_dir)
        self.results = []
    
    def load_tasks(self, task_file: str = "bench/tasks.md") -> List[Dict[str, Any]]:
        tasks = []
        
        if not Path(task_file).exists():
            print(f"Task file {task_file} not found, using default tasks")
            return self._get_default_tasks()
        
        with open(task_file, 'r') as f:
            content = f.read()
        
        current_task = {}
        for line in content.split('\n'):
            if line.startswith('## Task:'):
                if current_task:
                    tasks.append(current_task)
                current_task = {'name': line.replace('## Task:', '').strip()}
            elif line.startswith('- Query:'):
                current_task['query'] = line.replace('- Query:', '').strip()
            elif line.startswith('- Schema:'):
                current_task['schema_id'] = line.replace('- Schema:', '').strip()
            elif line.startswith('- URL:'):
                current_task['url'] = line.replace('- URL:', '').strip()
            elif line.startswith('- Selector:'):
                current_task['selector'] = line.replace('- Selector:', '').strip()
            elif line.startswith('- Expected:'):
                current_task['expected'] = line.replace('- Expected:', '').strip()
        
        if current_task:
            tasks.append(current_task)
        
        return tasks if tasks else self._get_default_tasks()
    
    def _get_default_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Product Search",
                "query": "Find MacBook Pro 16 inch price and availability",
                "schema_id": "product_v1",
                "expected": "product with price"
            },
            {
                "name": "Blog Post Analysis",
                "query": "Get latest blog post about machine learning with author info",
                "schema_id": "post_v1",
                "expected": "post with title and author"
            },
            {
                "name": "Event Query",
                "query": "Tech conference details for next month in San Francisco",
                "schema_id": "event_v1",
                "expected": "event with date and location"
            },
            {
                "name": "Product with URL",
                "query": "Extract product from webpage",
                "url": "https://store.example.com/item/123",
                "selector": ".product-details",
                "schema_id": "product_v1",
                "expected": "product from webpage"
            },
            {
                "name": "Complex Blog Query",
                "query": "Find all blog posts about Python programming from last week",
                "schema_id": "post_v1",
                "expected": "posts with Python tag"
            }
        ]
    
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Running: {task['name']}")
        
        start_time = time.time()
        
        result = self.engine.run(
            query=task['query'],
            wants_json=True,
            url=task.get('url'),
            selector=task.get('selector'),
            schema_id=task.get('schema_id')
        )
        
        elapsed = time.time() - start_time
        
        task_result = {
            'task_name': task['name'],
            'query': task['query'],
            'schema_id': task.get('schema_id'),
            'success': result['ok'],
            'path': result['path'],
            'fixed': result.get('fixed', False),
            'latency_s': elapsed,
            'compression_ratio': 0,
            'fidelity': 0,
            'output': result.get('json'),
            'notes': result.get('notes', '')
        }
        
        if result['ok'] and result.get('json'):
            codes = self.engine.tokenizer.encode(task['query'])
            decoded = self.engine.tokenizer.decode(codes)
            task_result['compression_ratio'] = self.engine.tokenizer.compress_ratio(task['query'])
            task_result['fidelity'] = self.engine.tokenizer.fidelity(task['query'])
        
        return task_result
    
    def run_benchmark(self, tasks: List[Dict[str, Any]], output_dir: str = "out"):
        print(f"Running benchmark with {len(tasks)} tasks...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] ", end="")
            result = self.run_task(task)
            self.results.append(result)
            
            status = "✓" if result['success'] else "✗"
            print(f" {status} ({result['latency_s']:.3f}s, {result['path']})")
        
        self._generate_report(output_dir)
    
    def _generate_report(self, output_dir: str):
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tasks': len(self.results),
            'successful_tasks': sum(1 for r in self.results if r['success']),
            'failed_tasks': sum(1 for r in self.results if not r['success']),
            'metrics': {
                'avg_latency_s': sum(r['latency_s'] for r in self.results) / len(self.results),
                'min_latency_s': min(r['latency_s'] for r in self.results),
                'max_latency_s': max(r['latency_s'] for r in self.results),
                'avg_compression': sum(r['compression_ratio'] for r in self.results) / len(self.results),
                'avg_fidelity': sum(r['fidelity'] for r in self.results) / len(self.results),
                'fast_path_usage': sum(1 for r in self.results if r['path'] == 'FAST_PATH') / len(self.results),
                'fixes_applied': sum(1 for r in self.results if r['fixed'])
            },
            'results': self.results
        }
        
        report_file = Path(output_dir) / 'report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Benchmark Report")
        print("=" * 60)
        print(f"Success Rate: {report['successful_tasks']}/{report['total_tasks']} "
              f"({report['successful_tasks']/report['total_tasks']*100:.1f}%)")
        print(f"Average Latency: {report['metrics']['avg_latency_s']:.3f}s")
        print(f"Latency Range: {report['metrics']['min_latency_s']:.3f}s - "
              f"{report['metrics']['max_latency_s']:.3f}s")
        print(f"Compression Ratio: {report['metrics']['avg_compression']:.2f}x")
        print(f"Fidelity: {report['metrics']['avg_fidelity']*100:.1f}%")
        print(f"Fast Path Usage: {report['metrics']['fast_path_usage']*100:.1f}%")
        print(f"Fixes Applied: {report['metrics']['fixes_applied']}")
        print(f"\nFull report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Run NeuroTiny benchmarks')
    parser.add_argument('--tasks', default='bench/tasks.md', help='Task file path')
    parser.add_argument('--output', default='out', help='Output directory')
    parser.add_argument('--checkpoints', default='checkpoints', help='Checkpoint directory')
    args = parser.parse_args()
    
    runner = BenchmarkRunner(checkpoint_dir=args.checkpoints)
    tasks = runner.load_tasks(args.tasks)
    runner.run_benchmark(tasks, args.output)


if __name__ == "__main__":
    main()