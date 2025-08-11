from typing import Dict, Any, List, Optional, Set
from enum import Enum
import json


class JSONState(Enum):
    START = "start"
    OBJECT_START = "object_start"
    OBJECT_KEY = "object_key"
    OBJECT_COLON = "object_colon"
    OBJECT_VALUE = "object_value"
    OBJECT_COMMA = "object_comma"
    ARRAY_START = "array_start"
    ARRAY_VALUE = "array_value"
    ARRAY_COMMA = "array_comma"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    NULL = "null"
    END = "end"


class JSONGrammarConstraint:
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema
        self.state_stack = [JSONState.START]
        self.current_path = []
        self.buffer = ""
        self.completed_fields = set()
    
    def get_valid_tokens(self, current_state: str = "") -> List[str]:
        if not self.state_stack:
            return []
        
        state = self.state_stack[-1]
        valid = []
        
        if state == JSONState.START:
            valid = ['{', '[', '"']
            if self.schema and self.schema.get('type') == 'object':
                valid = ['{']
            elif self.schema and self.schema.get('type') == 'array':
                valid = ['[']
        
        elif state == JSONState.OBJECT_START:
            if self.schema:
                required_fields = self._get_required_fields()
                available_fields = required_fields - self.completed_fields
                if available_fields:
                    field = min(available_fields)
                    valid = [f'"{field}"']
                else:
                    valid = ['}']
            else:
                valid = ['"', '}']
        
        elif state == JSONState.OBJECT_KEY:
            valid = [':']
        
        elif state == JSONState.OBJECT_COLON:
            valid = self._get_value_start_tokens()
        
        elif state == JSONState.OBJECT_VALUE:
            valid = [',', '}']
        
        elif state == JSONState.OBJECT_COMMA:
            if self.schema:
                required_fields = self._get_required_fields()
                available_fields = required_fields - self.completed_fields
                if available_fields:
                    field = min(available_fields)
                    valid = [f'"{field}"']
                else:
                    valid = ['}']
            else:
                valid = ['"']
        
        elif state == JSONState.ARRAY_START:
            valid = self._get_value_start_tokens() + [']']
        
        elif state == JSONState.ARRAY_VALUE:
            valid = [',', ']']
        
        elif state == JSONState.ARRAY_COMMA:
            valid = self._get_value_start_tokens()
        
        elif state == JSONState.STRING:
            valid = ['any_char', '"']
        
        elif state == JSONState.NUMBER:
            valid = ['0-9', '.', 'e', 'E', '+', '-', ',', '}', ']']
        
        elif state == JSONState.BOOLEAN:
            if 'true' in current_state.lower():
                valid = ['e'] if len(current_state) < 4 else [',', '}', ']']
            elif 'false' in current_state.lower():
                valid = ['e'] if len(current_state) < 5 else [',', '}', ']']
            else:
                valid = ['t', 'f']
        
        elif state == JSONState.NULL:
            valid = ['null'] if not current_state else [',', '}', ']']
        
        return valid
    
    def _get_value_start_tokens(self) -> List[str]:
        if self.schema and self.current_path:
            field_schema = self._get_field_schema()
            if field_schema:
                field_type = field_schema.get('type')
                if field_type == 'string':
                    return ['"']
                elif field_type == 'number':
                    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']
                elif field_type == 'boolean':
                    return ['t', 'f']
                elif field_type == 'array':
                    return ['[']
                elif field_type == 'object':
                    return ['{']
                elif field_type == 'null':
                    return ['n']
        
        return ['{', '[', '"', '0-9', '-', 't', 'f', 'n']
    
    def _get_field_schema(self) -> Optional[Dict[str, Any]]:
        if not self.schema or not self.current_path:
            return None
        
        schema = self.schema
        for part in self.current_path:
            if schema.get('type') == 'object':
                schema = schema.get('properties', {}).get(part)
            elif schema.get('type') == 'array':
                schema = schema.get('items')
            
            if not schema:
                return None
        
        return schema
    
    def _get_required_fields(self) -> Set[str]:
        if not self.schema:
            return set()
        
        if self.current_path:
            field_schema = self._get_field_schema()
            if field_schema and field_schema.get('type') == 'object':
                return set(field_schema.get('required', []))
        else:
            return set(self.schema.get('required', []))
        
        return set()
    
    def transition(self, token: str) -> bool:
        if not self.state_stack:
            return False
        
        state = self.state_stack[-1]
        
        if state == JSONState.START:
            if token == '{':
                self.state_stack.append(JSONState.OBJECT_START)
                self.buffer += token
                return True
            elif token == '[':
                self.state_stack.append(JSONState.ARRAY_START)
                self.buffer += token
                return True
        
        elif state == JSONState.OBJECT_START:
            if token.startswith('"'):
                field = token.strip('"')
                self.current_path.append(field)
                self.state_stack.append(JSONState.OBJECT_KEY)
                self.buffer += token
                return True
            elif token == '}':
                self.state_stack.pop()
                self.buffer += token
                return True
        
        elif state == JSONState.OBJECT_KEY:
            if token == ':':
                self.state_stack.pop()
                self.state_stack.append(JSONState.OBJECT_COLON)
                self.buffer += token
                return True
        
        elif state == JSONState.OBJECT_COLON:
            self.state_stack.pop()
            self.state_stack.append(JSONState.OBJECT_VALUE)
            
            if token == '{':
                self.state_stack.append(JSONState.OBJECT_START)
            elif token == '[':
                self.state_stack.append(JSONState.ARRAY_START)
            elif token == '"':
                self.state_stack.append(JSONState.STRING)
            elif token in '0123456789-':
                self.state_stack.append(JSONState.NUMBER)
            elif token in 'tf':
                self.state_stack.append(JSONState.BOOLEAN)
            elif token == 'n':
                self.state_stack.append(JSONState.NULL)
            
            self.buffer += token
            return True
        
        return False
    
    def is_complete(self) -> bool:
        return len(self.state_stack) == 1 and self.state_stack[0] == JSONState.END
    
    def reset(self):
        self.state_stack = [JSONState.START]
        self.current_path = []
        self.buffer = ""
        self.completed_fields = set()


class ConstrainedJSONDecoder:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.grammar = JSONGrammarConstraint(schema)
    
    def decode_step(self, partial: str, next_token: str) -> Tuple[bool, str]:
        valid_tokens = self.grammar.get_valid_tokens(partial)
        
        if next_token in valid_tokens:
            if self.grammar.transition(next_token):
                return True, partial + next_token
        
        return False, partial
    
    def validate_partial(self, partial: str) -> bool:
        try:
            json.loads(partial)
            return True
        except:
            depth = partial.count('{') - partial.count('}')
            depth += partial.count('[') - partial.count(']')
            return depth >= 0
    
    def suggest_completion(self, partial: str) -> str:
        depth_obj = partial.count('{') - partial.count('}')
        depth_arr = partial.count('[') - partial.count(']')
        
        completion = ""
        
        if partial.endswith(':'):
            completion = '""'
        elif partial.endswith('['):
            completion = ']'
        elif partial.endswith('{'):
            completion = '}'
        
        completion += '}' * depth_obj
        completion += ']' * depth_arr
        
        return completion