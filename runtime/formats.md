# NeuroTiny Runtime Formats

## Plan Format
Plans are multi-step descriptions of how to accomplish a task. Each step is numbered and describes a specific action.

### Template:
```
1. [Action verb] [target/resource]
2. [Process/Transform] [data]
3. [Generate/Format] [output]
```

### Examples:
```
1. Fetch data from https://api.example.com/products
2. Extract price and availability fields
3. Format as ProductV1 JSON
```

```
1. Query webpage https://example.com/events
2. Select elements with .event-item selector
3. Parse date, time, and location
4. Structure as EventV1 schema
```

## Action Format
Actions are specific operations to be performed, typically involving tools.

### Types:
- `FETCH <url>` - Retrieve content from URL
- `SELECT <selector>` - Extract elements using CSS/XPath selector
- `PARSE <field>` - Extract specific field from content
- `TRANSFORM <operation>` - Apply transformation to data
- `VALIDATE <schema>` - Check against schema

### Examples:
```
FETCH https://api.example.com/data
SELECT .product-price
PARSE date_field
TRANSFORM normalize_price
VALIDATE product_v1
```

## Observation Format
Observations are the results of actions, containing extracted or fetched data.

### Structure:
```json
{
  "source": "url or action",
  "timestamp": "ISO-8601 datetime",
  "data": {
    "field1": "value1",
    "field2": "value2"
  },
  "metadata": {
    "status": 200,
    "content_type": "text/html"
  }
}
```

## Check Format
Validation checks performed on generated JSON.

### Structure:
```json
{
  "schema_id": "product_v1",
  "valid": true/false,
  "errors": [
    "Missing required field: price",
    "Invalid format for field: date"
  ],
  "evidence_match": true/false
}
```

## Fix Format
Repairs applied to invalid JSON.

### Structure:
```json
{
  "action": "add_field|remove_field|fix_type|fix_format",
  "field": "field_name",
  "old_value": "previous_value",
  "new_value": "corrected_value",
  "reason": "Missing required field"
}
```

## Trace Format
Complete execution trace for training/debugging.

### Structure:
```json
{
  "id": "trace_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "query": "Original user query",
  "plan": "1. Step one\n2. Step two\n3. Step three",
  "actions": [
    {"type": "FETCH", "target": "url", "result": "..."}
  ],
  "observations": {
    "extracted_data": {},
    "evidence": {}
  },
  "json_output": {},
  "schema_id": "product_v1",
  "validation": {
    "valid": true,
    "errors": [],
    "fixes_applied": []
  },
  "metrics": {
    "latency_ms": 234,
    "compression_ratio": 2.3,
    "fidelity": 0.995,
    "path": "SLOW_PATH"
  }
}
```