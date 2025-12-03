"""
Simple script to start the API server
"""

import uvicorn
from api_server import app

if __name__ == "__main__":
    print("Starting Prompt Optimizer API Server...")
    print("Server will be available at http://127.0.0.1:8000")
    print("API docs available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")



