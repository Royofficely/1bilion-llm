#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import unittest
from unittest.mock import Mock, patch
from neurotok.api import NeuroTokenizer
from experts.reason_mini import ReasonMini
from experts.struct_mini import StructMini
from router.router import Router
from verify.schema_check import SchemaValidator
from verify.repair import JSONRepair
from runtime.engine import RuntimeEngine


class TestNeuroTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = NeuroTokenizer()
    
    def test_encode_decode(self):
        text = "Hello, world!"
        codes = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(codes)
        
        self.assertIsInstance(codes, list)
        self.assertIsInstance(decoded, str)
        self.assertTrue(len(codes) > 0)
    
    def test_compression_ratio(self):
        text = "This is a test string for compression."
        ratio = self.tokenizer.compress_ratio(text)
        
        self.assertIsInstance(ratio, float)
        self.assertGreater(ratio, 0)
    
    def test_fidelity(self):
        text = "Test fidelity calculation"
        fidelity = self.tokenizer.fidelity(text)
        
        self.assertIsInstance(fidelity, float)
        self.assertGreaterEqual(fidelity, 0)
        self.assertLessEqual(fidelity, 1)


class TestExperts(unittest.TestCase):
    def setUp(self):
        self.planner = ReasonMini()
        self.structurer = StructMini()
    
    def test_reason_mini_generate_plan(self):
        codes = [1, 2, 3, 4, 5]
        meta = {'query': 'test query', 'url': 'https://example.com'}
        
        plan = self.planner.generate_plan(codes, meta)
        
        self.assertIsInstance(plan, str)
        self.assertTrue('1.' in plan)
    
    def test_struct_mini_generate_json(self):
        observations = {
            'name': 'Test Product',
            'price': '99.99',
            'in_stock': 'true'
        }
        
        json_str = self.structurer.generate_json('product_v1', observations)
        json_data = json.loads(json_str)
        
        self.assertIsInstance(json_data, dict)
        self.assertIn('name', json_data)
        self.assertIn('price', json_data)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.router = Router()
    
    def test_fast_path_routing(self):
        meta = {
            'query': 'simple query',
            'wants_json': True,
            'url': None
        }
        
        decision = self.router.route(meta)
        
        self.assertEqual(decision.path, 'FAST_PATH')
        self.assertIn('struct_mini', decision.experts)
    
    def test_slow_path_routing(self):
        meta = {
            'query': 'complex query with multiple steps',
            'wants_json': True,
            'url': 'https://example.com'
        }
        
        decision = self.router.route(meta)
        
        self.assertEqual(decision.path, 'SLOW_PATH')
        self.assertTrue(len(decision.experts) > 1)


class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SchemaValidator()
    
    def test_valid_product_schema(self):
        valid_data = {
            "name": "Test Product",
            "price": 99.99,
            "currency": "USD",
            "in_stock": True,
            "url": "https://example.com/product"
        }
        
        is_valid, errors = self.validator.validate_json(valid_data, 'product_v1')
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_invalid_product_schema(self):
        invalid_data = {
            "name": "Test Product",
            "price": "invalid_price"
        }
        
        is_valid, errors = self.validator.validate_json(invalid_data, 'product_v1')
        
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)


class TestJSONRepair(unittest.TestCase):
    def setUp(self):
        self.repairer = JSONRepair()
    
    def test_repair_missing_field(self):
        incomplete_data = {
            "name": "Test Product",
            "price": 99.99
        }
        
        success, repaired, note = self.repairer.repair(incomplete_data, 'product_v1')
        
        if success:
            self.assertIn('currency', repaired)
            self.assertIn('in_stock', repaired)
            self.assertIn('url', repaired)


class TestRuntimeEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RuntimeEngine()
    
    def test_simple_query(self):
        result = self.engine.run(
            query="Test product",
            wants_json=True,
            schema_id='product_v1'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('ok', result)
        self.assertIn('json', result)
        self.assertIn('path', result)
    
    def test_with_url(self):
        with patch('tools.web_adapter.sync_fetch') as mock_fetch:
            mock_fetch.return_value = {
                'success': True,
                'data': [{'text': 'Test content'}],
                'provenance': {'url': 'https://example.com'}
            }
            
            result = self.engine.run(
                query="Extract data",
                wants_json=True,
                url="https://example.com",
                schema_id='product_v1'
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('ok', result)


def run_tests():
    print("Running NeuroTiny unit tests...")
    print("=" * 50)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestNeuroTokenizer,
        TestExperts,
        TestRouter,
        TestSchemaValidator,
        TestJSONRepair,
        TestRuntimeEngine
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")
    
    if result.errors:
        print("\nErrors:")
        for test, error in result.errors:
            print(f"- {test}: {error}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResult: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    run_tests()