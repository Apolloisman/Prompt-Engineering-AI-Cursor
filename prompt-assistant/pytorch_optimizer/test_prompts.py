"""Test the Neuro-Latent Optimizer with various prompts"""
import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def test_prompt(raw_prompt):
    """Test a single prompt"""
    print(f"\n{'='*70}")
    print(f"Testing: '{raw_prompt}'")
    print('='*70)
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/optimize",
            json={
                "raw_prompt": raw_prompt,
                "max_length": 300
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Original:  {raw_prompt}")
            print(f"Optimized: {result['optimized_prompt']}")
            print(f"Confidence: {result['confidence']:.2f}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test prompts
test_prompts = [
    "Create a function",
    "Build a website",
    "Analyze data",
    "Automate workflow",
    "Explain machine learning"
]

print("="*70)
print("NEURO-LATENT OPTIMIZER - PROMPT TESTING")
print("="*70)

for prompt in test_prompts:
    test_prompt(prompt)

print("\n" + "="*70)
print("Testing complete!")
print("="*70)


