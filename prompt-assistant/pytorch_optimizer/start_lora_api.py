"""
Startup script for Pure ML LoRA API Server
"""

import uvicorn
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print("Starting Pure ML Prompt Optimizer API Server...")
    print("Model: Flan-T5-base + LoRA")
    print("Approach: 100% Neural Network - No rule-based fallbacks")
    print("Server will be available at http://127.0.0.1:8000")
    print("API docs available at http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api_server_lora:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


