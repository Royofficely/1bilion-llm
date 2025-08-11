#!/usr/bin/env python3

import json
import random
from datetime import datetime, timedelta

def generate_traces(num_traces=50):
    traces = []
    
    product_queries = [
        "Get Samsung Galaxy S24 price",
        "Check AirPods availability",
        "Find PlayStation 5 in stock",
        "Dell XPS laptop specifications",
        "iPad Pro pricing and models",
        "Gaming chair under $300",
        "Best wireless headphones",
        "4K monitor deals",
        "Graphics card RTX 4090",
        "Smart watch comparison"
    ]
    
    post_queries = [
        "Latest blog about machine learning",
        "Python tips and tricks article",
        "Cloud computing trends post",
        "JavaScript framework comparison",
        "DevOps best practices guide",
        "Cybersecurity news update",
        "Data science tutorial",
        "Web development trends 2024",
        "Mobile app development guide",
        "AI ethics discussion"
    ]
    
    event_queries = [
        "Tech summit next month",
        "Developer conference schedule",
        "AI workshop registration",
        "Hackathon this weekend",
        "Webinar on cloud architecture",
        "Startup pitch event",
        "Coding bootcamp info session",
        "Virtual reality expo",
        "Data science meetup",
        "Open source contributors summit"
    ]
    
    schemas = ["product_v1", "post_v1", "event_v1"]
    paths = ["FAST_PATH", "SLOW_PATH"]
    
    for i in range(num_traces):
        schema = random.choice(schemas)
        path = random.choice(paths)
        
        if schema == "product_v1":
            query = random.choice(product_queries)
            json_output = {
                "name": f"Product {i}",
                "price": round(random.uniform(50, 2000), 2),
                "currency": "USD",
                "in_stock": random.choice([True, False]),
                "url": f"https://example.com/product-{i}"
            }
        elif schema == "post_v1":
            query = random.choice(post_queries)
            json_output = {
                "title": f"Blog Post {i}",
                "author": f"Author {random.randint(1, 10)}",
                "date": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
                "content": f"Content for post {i}...",
                "tags": random.sample(["Tech", "AI", "Python", "Cloud", "DevOps", "Data"], k=random.randint(1, 3))
            }
        else:
            query = random.choice(event_queries)
            json_output = {
                "name": f"Event {i}",
                "date": (datetime.now() + timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
                "time": f"{random.randint(9, 18):02d}:00",
                "location": random.choice(["Online", "San Francisco", "New York", "Austin", "Seattle"]),
                "description": f"Description for event {i}"
            }
        
        trace = {
            "id": f"trace_{i+1:03d}",
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 1000))).isoformat() + "Z",
            "query": query,
            "plan": f"1. Step one\n2. Step two\n3. Step three",
            "actions": [
                {"type": random.choice(["FETCH", "PARSE", "SELECT"]), 
                 "target": "data", 
                 "result": "processed"}
            ],
            "observations": {
                "extracted_data": {"field": "value"},
                "evidence": {"source_url": f"https://example.com/{i}"}
            },
            "json_output": json_output,
            "schema_id": schema,
            "validation": {
                "valid": random.choice([True, True, True, False]),
                "errors": [] if random.random() > 0.2 else ["Minor validation issue"],
                "fixes_applied": [] if random.random() > 0.1 else ["Applied minor fix"]
            },
            "metrics": {
                "latency_ms": random.randint(100, 1000),
                "compression_ratio": round(random.uniform(1.5, 3.0), 2),
                "fidelity": round(random.uniform(0.95, 0.999), 3),
                "path": path
            }
        }
        
        traces.append(trace)
    
    with open("data/traces/all_traces.jsonl", "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")
    
    print(f"Generated {num_traces} traces in data/traces/all_traces.jsonl")

if __name__ == "__main__":
    generate_traces(50)