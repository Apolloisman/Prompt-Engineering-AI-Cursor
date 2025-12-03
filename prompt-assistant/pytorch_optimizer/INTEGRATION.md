# PyTorch Latent Prompt Optimizer - Integration Guide

## Overview

The PyTorch Latent Prompt Optimizer is now **integrated** with your VS Code extension! When you use the prompt assistant, it will automatically use the PyTorch model if available, otherwise it falls back to the lightweight transformers.js approach.

## How It Works

### Architecture

```
User Prompt → PromptEnhancer → PyTorch Optimizer API → Latent Space Model → Optimized Prompt
                                    ↓ (if unavailable)
                              transformers.js fallback
```

### Integration Flow

1. **User enters prompt** in VS Code
2. **PromptEnhancer** checks if PyTorch API is available
3. **If available**: Sends prompt to PyTorch Latent Prompt Optimizer
   - Model encodes prompt → applies Delta Network → decodes optimized prompt
   - Returns ideal prompt with confidence score
4. **If unavailable**: Falls back to transformers.js-based enhancement
5. **Result**: Enhanced prompt displayed to user

## Setup Instructions

### Step 1: Install Python Dependencies

```bash
cd prompt-assistant/pytorch_optimizer
pip install -r requirements.txt
```

This installs:
- PyTorch
- Transformers (RoBERTa-large, GPT-2)
- FastAPI (API server)
- Uvicorn (ASGI server)

### Step 2: Start the API Server

**Option A: Run directly**
```bash
cd prompt-assistant/pytorch_optimizer
python start_api.py
```

**Option B: Run with uvicorn**
```bash
cd prompt-assistant/pytorch_optimizer
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

The server will:
- Load the PyTorch model (~1.5GB download on first run)
- Start on `http://127.0.0.1:8000`
- Provide API endpoints for prompt optimization

### Step 3: Use the Extension

1. Open VS Code/Cursor
2. The extension automatically detects the API server
3. Use the prompt assistant as normal - it will use PyTorch optimizer automatically!

## API Endpoints

### `POST /optimize`
Optimize a single prompt.

**Request:**
```json
{
  "raw_prompt": "Create a website",
  "rule_indices": [0, 1, 2],
  "return_embeddings": false
}
```

**Response:**
```json
{
  "optimized_prompt": "Create a modern, responsive website...",
  "confidence": 0.85,
  "raw_embedding": [...],
  "optimized_embedding": [...]
}
```

### `POST /feedback`
Submit user feedback to improve the model.

**Request:**
```json
{
  "raw_prompt": "Create a function",
  "edited_prompt": "Create a Python function with input validation...",
  "success_score": 0.9
}
```

### `GET /health`
Check if the API server is running and model is loaded.

## Training the Model

### Step 1: Create Training Data

```bash
python train.py --create_sample --data_path ./data/training_data.json
```

### Step 2: Train the Model

```bash
python train.py --data_path ./data/training_data.json --num_epochs 10 --batch_size 8
```

### Step 3: Model Checkpoints

Trained models are saved to `checkpoints/best_model.pt`. The API server automatically loads this if available.

## How the Model Works

### Latent Space Architecture

1. **Encoder (RoBERTa-large)**: Converts raw prompt text → latent embedding `z`
2. **Delta Network (Black Box)**: Transforms `z` → optimized embedding `z'`
3. **Decoder (GPT-2)**: Reconstructs `z'` → optimized prompt text

### Rule Seeding

The model starts with 8 heuristic rules:
- Rule 0: Be Specific
- Rule 1: Add Context
- Rule 2: Define Output Format
- Rule 3: Include Constraints
- Rule 4: Set Success Criteria
- Rule 5: Add Examples
- Rule 6: Specify Domain
- Rule 7: Structure Steps

These are injected as "Control Embeddings" that guide the initial optimization. As the model trains, it learns to prioritize its own learned patterns.

### Contrastive Learning

The model learns from triplets:
- **Anchor**: Raw prompt
- **Positive**: Successful/optimized prompt
- **Negative**: Failed/vague prompt

The Delta Network learns to move anchor embeddings closer to positive embeddings and away from negative embeddings.

## Feedback Integration

When users edit prompts in VS Code, the extension can submit feedback:

```typescript
await pytorchClient.submitFeedback({
    raw_prompt: "Create a function",
    edited_prompt: "Create a Python function with input validation...",
    success_score: 0.9
});
```

This updates the model weights to learn from user preferences.

## Troubleshooting

### API Server Not Starting

1. Check Python version: `python --version` (needs 3.8+)
2. Install dependencies: `pip install -r requirements.txt`
3. Check port 8000 is available: `netstat -an | grep 8000`

### Model Not Loading

1. First run downloads ~1.5GB of models (RoBERTa-large + GPT-2)
2. Check internet connection
3. Check disk space (~2GB needed)

### Extension Not Using PyTorch

1. Check API server is running: `curl http://127.0.0.1:8000/health`
2. Check extension logs in VS Code Output panel
3. Extension automatically falls back to transformers.js if API unavailable

## Performance

- **First optimization**: ~2-3 seconds (model loading)
- **Subsequent optimizations**: ~0.5-1 second
- **Memory usage**: ~2-3GB RAM
- **GPU**: Optional but recommended for faster inference

## Next Steps

1. **Train on your data**: Collect prompt pairs and train the model
2. **Fine-tune rules**: Adjust rule embeddings for your use case
3. **Collect feedback**: Use feedback integration to improve over time
4. **Monitor performance**: Track confidence scores and user satisfaction

## API Documentation

Full API docs available at: `http://127.0.0.1:8000/docs` (when server is running)



