"""Test the Neuro-Latent Optimizer API"""
import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Test health endpoint
print("Testing API Server...")
try:
    health = requests.get("http://127.0.0.1:8000/health", timeout=5)
    print(f"Health Status: {health.json()}")
    
    # Test optimization
    print("\nTesting prompt optimization...")
    test_prompt = "Create a function"
    
    response = requests.post(
        "http://127.0.0.1:8000/optimize",
        json={
            "raw_prompt": test_prompt,
            "max_length": 200
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Optimization successful!")
        print(f"Original: {test_prompt}")
        print(f"Optimized: {result['optimized_prompt']}")
        print(f"Confidence: {result['confidence']:.2f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.RequestException as e:
    print(f"Error connecting to API: {e}")
    print("Make sure the API server is running: python start_neuro_api.py")


