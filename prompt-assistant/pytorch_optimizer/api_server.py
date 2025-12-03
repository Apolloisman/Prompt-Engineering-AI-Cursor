"""
FastAPI Server for Latent Prompt Optimizer
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
from latent_prompt_optimizer import LatentPromptOptimizer, ContrastivePromptTrainer

app = FastAPI(title="Prompt Optimizer API")

# Enable CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to VS Code extension origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[LatentPromptOptimizer] = None
trainer: Optional[ContrastivePromptTrainer] = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OptimizeRequest(BaseModel):
    raw_prompt: str
    rule_indices: Optional[List[int]] = None
    return_embeddings: bool = False

class OptimizeResponse(BaseModel):
    optimized_prompt: str
    raw_embedding: Optional[List[float]] = None
    optimized_embedding: Optional[List[float]] = None
    confidence: float

class FeedbackRequest(BaseModel):
    raw_prompt: str
    edited_prompt: str
    success_score: float  # 0.0 to 1.0
    rule_indices: Optional[List[int]] = None

class FeedbackResponse(BaseModel):
    success: bool
    message: str

class BatchOptimizeRequest(BaseModel):
    prompts: List[str]
    rule_indices: Optional[List[int]] = None

class BatchOptimizeResponse(BaseModel):
    optimized_prompts: List[str]
    confidences: List[float]

@app.on_event("startup")
async def load_model():
    """Load the PyTorch model on startup"""
    global model, trainer
    try:
        print(f"Loading model on device: {device}")
        model = LatentPromptOptimizer(device=device)
        
        # Try to load trained weights - check for best_model.pt first, then latest checkpoint
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        model_path = None
        
        # First try best_model.pt
        best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            # Find latest checkpoint_epoch_*.pt
            if os.path.exists(checkpoints_dir):
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) 
                                  if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
                if checkpoint_files:
                    # Sort by epoch number
                    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                    model_path = os.path.join(checkpoints_dir, checkpoint_files[0])
        
        if model_path and os.path.exists(model_path):
            print(f"Loading trained weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                # Old format, direct state dict
                model.load_state_dict(checkpoint)
            model.eval()
        else:
            print("No trained weights found, using initialized model")
            model.eval()
        
        trainer = ContrastivePromptTrainer(model=model, learning_rate=1e-4, margin=0.5)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_prompt(request: OptimizeRequest):
    """
    Optimize a single prompt using the Latent Prompt Optimizer
    
    Args:
        raw_prompt: The original prompt to optimize
        rule_indices: Optional list of rule indices to apply (0-7)
        return_embeddings: Whether to return embedding vectors
    
    Returns:
        Optimized prompt text and optional embeddings
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert rule_indices to tensor if provided
        # Ensure it's a 1D tensor (list of rule indices for a single prompt)
        rule_tensor = None
        if request.rule_indices:
            # Convert to tensor and ensure it's 1D
            rule_tensor = torch.tensor(request.rule_indices, device=device, dtype=torch.long)
            if rule_tensor.dim() == 0:
                rule_tensor = rule_tensor.unsqueeze(0)
        
        # Run optimization
        # Always decode text (the model will generate optimized prompt)
        with torch.no_grad():
            z, z_prime, decoded_text = model.forward(
                request.raw_prompt,
                rule_indices=rule_tensor,
                return_embeddings=True  # Always decode to get optimized text
            )
        
        # Extract optimized prompt (decoded text or fallback to original)
        optimized_prompt = decoded_text if decoded_text and decoded_text.strip() else request.raw_prompt
        
        # Calculate confidence based on embedding distance
        # Lower distance = higher confidence
        if request.return_embeddings:
            distance = torch.norm(z - z_prime).item()
            confidence = max(0.0, min(1.0, 1.0 - (distance / 10.0)))  # Normalize to 0-1
        else:
            confidence = 0.8  # Default confidence
        
        response = OptimizeResponse(
            optimized_prompt=optimized_prompt,
            confidence=confidence
        )
        
        if request.return_embeddings:
            response.raw_embedding = z.cpu().numpy().tolist()
            response.optimized_embedding = z_prime.cpu().numpy().tolist()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/optimize/batch", response_model=BatchOptimizeResponse)
async def optimize_batch(request: BatchOptimizeRequest):
    """Optimize multiple prompts in batch"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        optimized_prompts = []
        confidences = []
        
        rule_tensor = None
        if request.rule_indices:
            rule_tensor = torch.tensor(request.rule_indices, device=device)
        
        for prompt in request.prompts:
            with torch.no_grad():
                z, z_prime, decoded_texts = model.forward(
                    prompt,
                    rule_indices=rule_tensor,
                    return_embeddings=False
                )
            
            optimized_prompt = decoded_texts[0] if decoded_texts else prompt
            distance = torch.norm(z - z_prime).item()
            confidence = max(0.0, min(1.0, 1.0 - (distance / 10.0)))
            
            optimized_prompts.append(optimized_prompt)
            confidences.append(confidence)
        
        return BatchOptimizeResponse(
            optimized_prompts=optimized_prompts,
            confidences=confidences
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch optimization failed: {str(e)}")

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback to improve the model
    
    When a user edits a prompt, this updates the model weights
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not loaded")
    
    try:
        rule_tensor = None
        if request.rule_indices:
            rule_tensor = torch.tensor(request.rule_indices, device=device)
        
        # Update model with feedback
        loss = trainer.update_from_feedback(
            raw_prompt=request.raw_prompt,
            edited_prompt=request.edited_prompt,
            success_score=request.success_score
        )
        
        return FeedbackResponse(
            success=True,
            message="Feedback integrated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback integration failed: {str(e)}")

@app.get("/rules")
async def get_rules():
    """Get available rule indices and descriptions"""
    return {
        "rules": [
            {"index": 0, "name": "Be Specific", "description": "Add specific details and requirements"},
            {"index": 1, "name": "Add Context", "description": "Include relevant background information"},
            {"index": 2, "name": "Define Output Format", "description": "Specify desired output structure"},
            {"index": 3, "name": "Include Constraints", "description": "Add limitations and constraints"},
            {"index": 4, "name": "Set Success Criteria", "description": "Define measurable success metrics"},
            {"index": 5, "name": "Add Examples", "description": "Include concrete examples"},
            {"index": 6, "name": "Specify Domain", "description": "Clarify domain/technology context"},
            {"index": 7, "name": "Structure Steps", "description": "Break down into actionable steps"}
        ]
    }

if __name__ == "__main__":
    # Run on localhost:8000 by default
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

