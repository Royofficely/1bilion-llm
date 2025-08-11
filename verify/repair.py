import json
import re
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from schema_check import SchemaValidator


class JSONRepair:
    def __init__(self):
        self.validator = SchemaValidator()
        self.repair_strategies = {
            'missing_field': self._repair_missing_field,
            'extra_field': self._repair_extra_field,
            'invalid_type': self._repair_invalid_type,
            'invalid_format': self._repair_invalid_format
        }
    
    def repair(self, json_data: Union[str, dict], schema_id: str, 
              max_attempts: int = 1) -> Tuple[bool, Union[str, dict], str]:
        
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError:
                fixed_str = self._fix_json_syntax(json_data)
                try:
                    data = json.loads(fixed_str)
                except:
                    return False, json_data, "Failed to parse JSON"
        else:
            data = json_data.copy()
        
        is_valid, errors = self.validator.validate_json(data, schema_id)
        
        if is_valid:
            return True, data, "Already valid"
        
        suggestions = self.validator.suggest_fixes(data, schema_id, errors)
        
        if not suggestions or max_attempts <= 0:
            return False, data, f"Cannot repair: {'; '.join(errors[:2])}"
        
        fix = suggestions[0]
        
        if fix['action'] == 'add_field':
            data[fix['field']] = fix['value']
            note = f"Added missing field '{fix['field']}'"
        
        elif fix['action'] == 'remove_field':
            data.pop(fix['field'], None)
            note = f"Removed extra field '{fix['field']}'"
        
        else:
            return False, data, "Unknown repair action"
        
        is_valid, _ = self.validator.validate_json(data, schema_id)
        
        return is_valid, data, note
    
    def _fix_json_syntax(self, json_str: str) -> str:
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        json_str = re.sub(r"'", '"', json_str)
        
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        return json_str
    
    def _repair_missing_field(self, data: dict, field: str, 
                              field_schema: Dict[str, Any]) -> Any:
        field_type = field_schema.get('type', 'string')
        
        if field_type == 'string':
            if 'default' in field_schema:
                return field_schema['default']
            elif 'format' in field_schema:
                if field_schema['format'] == 'date':
                    return datetime.now().strftime('%Y-%m-%d')
                elif field_schema['format'] == 'uri':
                    return 'https://example.com'
            return ''
        
        elif field_type == 'number':
            return field_schema.get('default', 0)
        
        elif field_type == 'boolean':
            return field_schema.get('default', False)
        
        elif field_type == 'array':
            return []
        
        elif field_type == 'object':
            return {}
        
        return None
    
    def _repair_extra_field(self, data: dict, field: str) -> dict:
        data_copy = data.copy()
        data_copy.pop(field, None)
        return data_copy
    
    def _repair_invalid_type(self, data: dict, field: str, 
                             expected_type: str) -> Any:
        value = data.get(field)
        
        if expected_type == 'string':
            return str(value) if value is not None else ''
        
        elif expected_type == 'number':
            try:
                if isinstance(value, str):
                    cleaned = re.sub(r'[^0-9.-]', '', value)
                    return float(cleaned)
                return float(value)
            except:
                return 0
        
        elif expected_type == 'boolean':
            if isinstance(value, str):
                return value.lower() in ['true', 'yes', '1']
            return bool(value)
        
        elif expected_type == 'array':
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                if ',' in value:
                    return [v.strip() for v in value.split(',')]
            return [value] if value else []
        
        return value
    
    def _repair_invalid_format(self, value: str, format_type: str) -> str:
        if format_type == 'date':
            date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', value)
            if date_match:
                date_str = date_match.group(0)
                if '/' in date_str:
                    parts = date_str.split('/')
                    if len(parts) == 3:
                        return f"{parts[2]}-{parts[0]:0>2}-{parts[1]:0>2}"
                return date_str
            return datetime.now().strftime('%Y-%m-%d')
        
        elif format_type == 'uri':
            if not value.startswith(('http://', 'https://')):
                return f"https://{value}"
            return value
        
        return value
    
    def minimal_repair(self, data: dict, schema_id: str) -> Tuple[dict, str]:
        suggestions = self.validator.suggest_fixes(data, schema_id, [])
        
        if not suggestions:
            return data, "No repairs needed"
        
        fix = suggestions[0]
        repaired = data.copy()
        
        if fix['action'] == 'add_field':
            repaired[fix['field']] = fix['value']
            return repaired, f"Added {fix['field']}"
        
        elif fix['action'] == 'remove_field':
            repaired.pop(fix['field'], None)
            return repaired, f"Removed {fix['field']}"
        
        return data, "No repair applied"


from typing import Union