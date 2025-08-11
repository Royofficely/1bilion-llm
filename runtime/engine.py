import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path

from neurotok.api import NeuroTokenizer
from experts.reason_mini import ReasonMini
from experts.struct_mini import StructMini
from router.router import Router
from tools.web_adapter import WebAdapter, sync_fetch
from tools.evidence import EvidenceNormalizer, EvidenceExtractor
from verify.schema_check import SchemaValidator
from verify.repair import JSONRepair
from constraints.json_grammar import ConstrainedJSONDecoder


class RuntimeEngine:
    def __init__(self, checkpoint_dir: Optional[str] = "checkpoints"):
        self.tokenizer = NeuroTokenizer(
            checkpoint_path=f"{checkpoint_dir}/neurotok.pt" if checkpoint_dir else None,
            device='cuda' if self._cuda_available() else 'cpu'
        )
        
        self.router = Router()
        self.planner = ReasonMini()
        self.structurer = StructMini()
        self.validator = SchemaValidator()
        self.repairer = JSONRepair()
        self.evidence_normalizer = EvidenceNormalizer()
        self.evidence_extractor = EvidenceExtractor()
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._load_checkpoints()
    
    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _load_checkpoints(self):
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return
        
        planner_ckpt = self.checkpoint_dir / "reason_mini.pt"
        if planner_ckpt.exists():
            self.planner.load_checkpoint(str(planner_ckpt))
        
        struct_ckpt = self.checkpoint_dir / "struct_mini.pt"
        if struct_ckpt.exists():
            self.structurer.load_checkpoint(str(struct_ckpt))
    
    def run(self, query: str, wants_json: bool = True, 
           url: Optional[str] = None, selector: Optional[str] = None,
           schema_id: Optional[str] = None) -> Dict[str, Any]:
        
        start_time = time.time()
        
        meta = {
            'query': query,
            'wants_json': wants_json,
            'url': url,
            'selector': selector,
            'schema_id': schema_id
        }
        
        codes = self.tokenizer.encode(query)
        
        route_decision = self.router.route(meta)
        
        result = {
            'ok': False,
            'fixed': False,
            'json': None,
            'notes': '',
            'latency_s': 0,
            'path': route_decision.path
        }
        
        try:
            if route_decision.path == 'FAST_PATH':
                result = self._fast_path(codes, meta, schema_id)
            else:
                result = self._slow_path(codes, meta, route_decision, schema_id)
            
            result['ok'] = True
            
        except Exception as e:
            result['notes'] = f"Error: {str(e)}"
        
        result['latency_s'] = time.time() - start_time
        result['path'] = route_decision.path
        
        return result
    
    def _fast_path(self, codes: List[int], meta: Dict[str, Any], 
                  schema_id: Optional[str]) -> Dict[str, Any]:
        
        if not schema_id:
            schema_id = self._infer_schema(meta['query'])
        
        observations = {'query': meta['query']}
        json_str = self.structurer.generate_json(schema_id, observations, codes)
        
        try:
            json_data = json.loads(json_str)
        except:
            json_data = {'error': 'Invalid JSON generated'}
        
        return {
            'ok': True,
            'fixed': False,
            'json': json_data,
            'notes': f'Fast path with {schema_id}',
            'latency_s': 0,
            'path': 'FAST_PATH'
        }
    
    def _slow_path(self, codes: List[int], meta: Dict[str, Any], 
                  route_decision: Any, schema_id: Optional[str]) -> Dict[str, Any]:
        
        plan = self.planner.generate_plan(codes, meta)
        
        observations = {'query': meta['query'], 'plan': plan}
        evidence = {}
        
        if meta.get('url'):
            fetch_result = sync_fetch(meta['url'], meta.get('selector'))
            
            if fetch_result['success']:
                observations['fetch_result'] = fetch_result.get('data', fetch_result.get('content', ''))
                evidence = self.evidence_normalizer.normalize(fetch_result)
                
                if 'data' in fetch_result:
                    extracted = self.evidence_extractor.extract_structured_data(fetch_result['data'])
                    observations.update(extracted)
        
        if not schema_id:
            schema_id = self._infer_schema(meta['query'])
        
        json_str = self.structurer.generate_json(schema_id, observations, codes)
        
        try:
            json_data = json.loads(json_str)
        except:
            json_data = {'error': 'Invalid JSON generated'}
            return {
                'ok': False,
                'fixed': False,
                'json': json_data,
                'notes': 'Failed to generate valid JSON',
                'latency_s': 0,
                'path': 'SLOW_PATH'
            }
        
        if 'verifier' in route_decision.experts:
            is_valid, errors = self.validator.validate_json(json_data, schema_id, evidence)
            
            if not is_valid:
                fixed, repaired_data, repair_note = self.repairer.repair(json_data, schema_id)
                
                if fixed:
                    json_data = repaired_data
                    notes = f"Repaired: {repair_note}"
                    return {
                        'ok': True,
                        'fixed': True,
                        'json': json_data,
                        'notes': notes,
                        'latency_s': 0,
                        'path': 'SLOW_PATH'
                    }
                else:
                    notes = f"Validation failed: {'; '.join(errors[:2])}"
            else:
                notes = "Validated successfully"
        else:
            notes = "No validation performed"
        
        return {
            'ok': True,
            'fixed': False,
            'json': json_data,
            'notes': notes,
            'latency_s': 0,
            'path': 'SLOW_PATH'
        }
    
    def _infer_schema(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['product', 'price', 'stock', 'buy', 'shop']):
            return 'product_v1'
        elif any(word in query_lower for word in ['post', 'article', 'blog', 'author', 'tag']):
            return 'post_v1'
        elif any(word in query_lower for word in ['event', 'date', 'time', 'location', 'venue']):
            return 'event_v1'
        
        return 'product_v1'


from typing import List


def main():
    engine = RuntimeEngine()
    
    result = engine.run(
        query="Get product info from example.com",
        wants_json=True,
        url="https://example.com/product",
        selector=".product-info",
        schema_id="product_v1"
    )
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()