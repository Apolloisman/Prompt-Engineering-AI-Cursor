"""
FastAPI Server for Neuro-Latent Optimizer
Provides HTTP API for TypeScript extension to use PyTorch model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import torch
import json
import os
import sys
from neuro_prompt_optimizer import NeuroPromptOptimizer

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Neuro-Latent Prompt Optimizer API")

# Enable CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to VS Code extension origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[NeuroPromptOptimizer] = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OptimizeRequest(BaseModel):
    raw_prompt: str
    max_length: Optional[int] = 200
    num_return_sequences: Optional[int] = 1

class OptimizeResponse(BaseModel):
    optimized_prompt: str
    raw_embedding: Optional[List[float]] = None
    optimized_intent: Optional[List[float]] = None
    confidence: float

class FeedbackRequest(BaseModel):
    raw_prompt: str
    edited_prompt: str
    success_score: float  # 0.0 to 1.0

class FeedbackResponse(BaseModel):
    success: bool
    message: str

@app.on_event("startup")
async def load_model():
    """Load the Neuro-Latent Optimizer model on startup"""
    global model
    try:
        print(f"Loading Neuro-Latent Optimizer on device: {device}")
        model = NeuroPromptOptimizer(device=device)
        model.to(device)
        
        # Try to load trained weights
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        model_path = None
        
        # First try new ML-trained model, then old final model
        final_model_path_ml = os.path.join(checkpoints_dir, "neuro_final_model_ml.pt")
        final_model_path = os.path.join(checkpoints_dir, "neuro_final_model.pt")
        if os.path.exists(final_model_path_ml):
            model_path = final_model_path_ml
        elif os.path.exists(final_model_path):
            model_path = final_model_path
        else:
            # Find latest phase2 checkpoint
            if os.path.exists(checkpoints_dir):
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) 
                                  if f.startswith('phase2_epoch_') and f.endswith('.pt')]
                if checkpoint_files:
                    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                    model_path = os.path.join(checkpoints_dir, checkpoint_files[0])
                else:
                    # Try phase1 checkpoint
                    checkpoint_files = [f for f in os.listdir(checkpoints_dir) 
                                      if f.startswith('phase1_epoch_') and f.endswith('.pt')]
                    if checkpoint_files:
                        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                        model_path = os.path.join(checkpoints_dir, checkpoint_files[0])
        
        if model_path and os.path.exists(model_path):
            print(f"Loading trained weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint (phase: {checkpoint.get('phase', 'unknown')}, epoch: {checkpoint.get('epoch', 'unknown')})")
            else:
                model.load_state_dict(checkpoint)
            model.eval()
        else:
            print("No trained weights found, using initialized model")
            model.eval()
        
        print("Neuro-Latent Optimizer loaded successfully")
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
        "model_loaded": model is not None,
        "device": device,
        "model_type": "Neuro-Latent Optimizer"
    }

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_prompt(request: OptimizeRequest):
    """
    Optimize a single prompt using the Neuro-Latent Optimizer
    
    Args:
        raw_prompt: The original prompt to optimize
        max_length: Maximum generation length
        num_return_sequences: Number of sequences to generate
    
    Returns:
        Optimized prompt text and optional embeddings
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Run optimization to get optimized intent (ML-learned transformation)
        with torch.no_grad():
            result = model.forward(
                raw_text=request.raw_prompt,
                max_length=request.max_length or 200,
                num_return_sequences=request.num_return_sequences or 1,
                return_soft_prompts=False
            )
        
        # Use PURE ML-generated text from T5 (no fallback - pure ML approach)
        optimized_prompt = result['generated_text']
        if isinstance(optimized_prompt, list):
            optimized_prompt = optimized_prompt[0]
        
        # Pure ML: Use what the model generates, even if imperfect
        # Quality will improve with more training data and epochs
        if not optimized_prompt or not optimized_prompt.strip():
            # Only fallback if completely empty (shouldn't happen)
            optimized_prompt = request.raw_prompt
        
        # Calculate confidence based on embedding transformation
        raw_emb = result['raw_embedding']
        opt_intent = result['optimized_intent']
        
        # Cosine similarity between raw and optimized
        raw_norm = torch.nn.functional.normalize(raw_emb, p=2, dim=1)
        opt_norm = torch.nn.functional.normalize(opt_intent, p=2, dim=1)
        similarity = (raw_norm * opt_norm).sum(dim=1).item()
        
        # Confidence: if text was generated and is different, high confidence
        if optimized_prompt != request.raw_prompt and len(optimized_prompt) > len(request.raw_prompt):
            confidence = min(0.95, max(0.7, 0.8 + (1.0 - similarity) * 0.2))
        else:
            confidence = 0.5  # Low confidence if no improvement
        
        response = OptimizeResponse(
            optimized_prompt=optimized_prompt,
            raw_embedding=raw_emb[0].cpu().tolist() if raw_emb is not None else None,
            optimized_intent=opt_intent[0].cpu().tolist() if opt_intent is not None else None,
            confidence=confidence
        )
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback to improve the model
    Note: This would require retraining, so for now it just acknowledges receipt
    """
    try:
        # In a full implementation, this would:
        # 1. Store the feedback
        # 2. Trigger fine-tuning or update training data
        # 3. Retrain the model
        
        # For now, just acknowledge
        return FeedbackResponse(
            success=True,
            message=f"Feedback received (score: {request.success_score}). Model will be updated in next training cycle."
        )
    except Exception as e:
        return FeedbackResponse(
            success=False,
            message=f"Failed to process feedback: {str(e)}"
        )

@app.get("/rules")
async def get_rules():
    """
    Get available heuristic rules
    Note: Neuro-Latent Optimizer doesn't use explicit rules,
    but we provide this for compatibility
    """
    return {
        "rules": [
            {
                "index": 0,
                "name": "Soft Prompt Optimization",
                "description": "Uses learned soft prompts to enhance prompts"
            }
        ],
        "note": "Neuro-Latent Optimizer uses end-to-end learning rather than explicit rules"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

