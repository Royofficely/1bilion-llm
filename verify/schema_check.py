import json
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path


class SchemaValidator:
    def __init__(self, schema_dir: str = "verify/schemas"):
        self.schema_dir = Path(schema_dir)
        self.schemas = {}
        self._load_schemas()
    
    def _load_schemas(self):
        if not self.schema_dir.exists():
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            schema_id = schema_file.stem
            try:
                with open(schema_file, 'r') as f:
                    self.schemas[schema_id] = json.load(f)
            except Exception as e:
                print(f"Failed to load schema {schema_id}: {e}")
    
    def validate_json(self, json_data: Union[str, dict], schema_id: str, 
                     evidence: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        errors = []
        
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return False, [f"Invalid JSON: {e}"]
        else:
            data = json_data
        
        if schema_id not in self.schemas:
            return False, [f"Unknown schema: {schema_id}"]
        
        schema = self.schemas[schema_id]
        
        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            
            if e.absolute_path:
                errors.append(f"Error at path: {'.'.join(str(p) for p in e.absolute_path)}")
        
        if evidence:
            evidence_errors = self._validate_with_evidence(data, schema_id, evidence)
            errors.extend(evidence_errors)
        
        return len(errors) == 0, errors
    
    def _validate_with_evidence(self, data: dict, schema_id: str, 
                                evidence: Dict[str, Any]) -> List[str]:
        errors = []
        
        if schema_id == 'product_v1':
            if 'url' in data and 'source_url' in evidence:
                if not self._urls_match(data['url'], evidence['source_url']):
                    errors.append(f"URL mismatch: {data['url']} != {evidence['source_url']}")
        
        elif schema_id == 'post_v1':
            if 'date' in data and 'timestamp' in evidence:
                pass
        
        elif schema_id == 'event_v1':
            if 'location' in data and 'extracted_data' in evidence:
                extracted = evidence.get('extracted_data', {})
                if 'location' in extracted and data['location'] != extracted['location']:
                    errors.append(f"Location mismatch with evidence")
        
        return errors
    
    def _urls_match(self, url1: str, url2: str) -> bool:
        url1 = url1.replace('http://', 'https://').rstrip('/')
        url2 = url2.replace('http://', 'https://').rstrip('/')
        return url1 == url2
    
    def get_missing_fields(self, data: dict, schema_id: str) -> List[str]:
        if schema_id not in self.schemas:
            return []
        
        schema = self.schemas[schema_id]
        required = schema.get('required', [])
        
        return [field for field in required if field not in data]
    
    def get_extra_fields(self, data: dict, schema_id: str) -> List[str]:
        if schema_id not in self.schemas:
            return []
        
        schema = self.schemas[schema_id]
        properties = schema.get('properties', {})
        
        return [field for field in data if field not in properties]
    
    def suggest_fixes(self, data: dict, schema_id: str, errors: List[str]) -> List[Dict[str, Any]]:
        suggestions = []
        
        missing_fields = self.get_missing_fields(data, schema_id)
        for field in missing_fields:
            schema = self.schemas[schema_id]
            field_schema = schema['properties'].get(field, {})
            
            if field_schema.get('type') == 'string':
                default = field_schema.get('default', '')
                suggestions.append({
                    'action': 'add_field',
                    'field': field,
                    'value': default
                })
            elif field_schema.get('type') == 'number':
                suggestions.append({
                    'action': 'add_field',
                    'field': field,
                    'value': 0
                })
            elif field_schema.get('type') == 'boolean':
                suggestions.append({
                    'action': 'add_field',
                    'field': field,
                    'value': False
                })
            elif field_schema.get('type') == 'array':
                suggestions.append({
                    'action': 'add_field',
                    'field': field,
                    'value': []
                })
        
        extra_fields = self.get_extra_fields(data, schema_id)
        for field in extra_fields:
            suggestions.append({
                'action': 'remove_field',
                'field': field
            })
        
        return suggestions


from typing import Union