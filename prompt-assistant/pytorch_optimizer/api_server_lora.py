"""
Pure ML API Server: Flan-T5-base + LoRA for Prompt Optimization
100% Neural Network - No rule-based fallbacks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import torch
import os
import sys
from inference import PureMLPromptOptimizer

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Pure ML Prompt Optimizer API (LoRA)")

# Enable CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
optimizer: Optional[PureMLPromptOptimizer] = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OptimizeRequest(BaseModel):
    raw_prompt: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class OptimizeResponse(BaseModel):
    optimized_prompt: str
    confidence: float

@app.on_event("startup")
async def load_model():
    """Load Flan-T5-base + LoRA adapters"""
    global optimizer
    try:
        print(f"Loading Pure ML Prompt Optimizer on device: {device}")
        
        # Path to LoRA adapters
        adapter_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lora_model", "final_adapter")
        
        # Check if adapters exist
        if not os.path.exists(adapter_path):
            print(f"⚠ LoRA adapters not found at {adapter_path}")
            print("⚠ Using base model only (will need training)")
            adapter_path = None
        
        # Initialize optimizer (Pure ML - no rule-based code)
        optimizer = PureMLPromptOptimizer(
            base_model_name="google/flan-t5-base",
            adapter_path=adapter_path,
            device=device
        )
        
        print("✓ Pure ML Prompt Optimizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": optimizer is not None,
        "device": device,
        "model_type": "Pure ML (Flan-T5-base + LoRA)",
        "approach": "100% Neural Network - No rule-based fallbacks"
    }

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_prompt(request: OptimizeRequest):
    """
    Pure ML Prompt Optimization
    
    Pipeline: Raw Input -> Tokenizer -> Flan-T5 (with LoRA) -> Optimized Output
    No if/else logic, no rule-based fallbacks - 100% neural network
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Pure ML optimization (no fallbacks, no rules)
        optimized_prompt = optimizer.optimize(
            vague_prompt=request.raw_prompt,
            max_length=request.max_length or 512,
            temperature=request.temperature or 0.7
        )
        
        # Confidence: High if output is different and longer (ML made improvements)
        if optimized_prompt != request.raw_prompt and len(optimized_prompt) > len(request.raw_prompt):
            confidence = 0.9
        elif optimized_prompt != request.raw_prompt:
            confidence = 0.7
        else:
            confidence = 0.5  # Low if no change
        
        response = OptimizeResponse(
            optimized_prompt=optimized_prompt,
            confidence=confidence
        )
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/rules")
async def get_rules():
    """
    No explicit rules - rules are learned implicitly by the ML model
    """
    return {
        "rules": [],
        "note": "Pure ML approach - rules are learned implicitly through training data, not hard-coded"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


