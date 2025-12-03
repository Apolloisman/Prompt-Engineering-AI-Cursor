"""
Startup script for Neuro-Latent Optimizer API Server
"""

import uvicorn
import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print("Starting Neuro-Latent Optimizer API Server...")
    print("Server will be available at http://127.0.0.1:8000")
    print("API docs available at http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api_server_neuro:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


