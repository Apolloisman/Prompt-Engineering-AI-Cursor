# Neuro-Latent Optimizer Architecture

## Overview

The **Neuro-Latent Optimizer** uses **Soft Prompting (Prefix Tuning)** combined with **Latent Space Contrastive Learning** to optimize prompts. Unlike the previous VAE-style approach, this architecture:

1. Uses **frozen pre-trained models** (RoBERTa encoder, T5 generator)
2. Learns **soft prompts** (ghost tokens) that enhance prompts
3. Trains only a small **black box network** and **projector**
4. Generates **actual improved prompt text** using T5

## Architecture Components

### 1. Encoder: RoBERTa-base (Frozen)
- **Purpose**: Encode raw prompts into embeddings
- **Output**: 768-dimensional embedding vector
- **Status**: Frozen (not trained)

### 2. Black Box Network (Trainable)
- **Architecture**: 3-layer Dense Network
  - `Linear(768 → 1024)` → `ReLU` → `Dropout(0.1)` → `Linear(1024 → 768)`
- **Purpose**: Learn complex transformations from raw → optimized intent
- **Status**: Trainable (core learning component)

### 3. Projector (Trainable)
- **Architecture**: `Linear(768 → soft_prompt_length * 512)`
- **Purpose**: Convert optimized intent into soft prompt tokens
- **Output**: `[Batch, 20, 512]` tensor (20 ghost tokens)
- **Status**: Trainable

### 4. Generator: T5-small (Frozen)
- **Purpose**: Generate enhanced prompt text
- **Input**: Soft prompts concatenated with original input embeddings
- **Status**: Frozen (not trained)

## Forward Pass Flow

```
Raw Prompt Text
    ↓
[RoBERTa Encoder] → Raw Embedding (768-dim)
    ↓
[Black Box Network] → Optimized Intent (768-dim)
    ↓
[Projector] → Soft Prompts [20, 512]
    ↓
[Concatenate with Input Embeddings]
    ↓
[T5 Generator] → Enhanced Prompt Text
```

## Training Phases

### Phase 1: Distillation
- **Objective**: Learn to match teacher/enhanced prompts
- **Loss**: MSE between optimized intent and teacher embedding
- **Purpose**: Ensure validity and prevent fallback

### Phase 2: Contrastive Learning
- **Objective**: Pull optimized prompts closer to successful examples, push away from failed examples
- **Loss**: InfoNCE loss on optimized intent vectors
- **Purpose**: Learn what makes prompts successful

### Safety Mechanism: KL Divergence Penalty
- **Purpose**: Keep soft prompts within readable distribution
- **Prevents**: Gibberish generation
- **Weight**: 0.01 (configurable)

## Usage

### Training

```bash
# Create sample dataset
python train_neuro.py --create_sample --data_path ./data/neuro_training_data.json

# Train Phase 1 (Distillation)
python train_neuro.py --data_path ./data/neuro_training_data.json --phase1_epochs 3 --phase2_epochs 0

# Train Phase 2 (Contrastive)
python train_neuro.py --data_path ./data/neuro_training_data.json --phase1_epochs 0 --phase2_epochs 3

# Train both phases
python train_neuro.py --data_path ./data/neuro_training_data.json --phase1_epochs 3 --phase2_epochs 3
```

### API Server

```bash
# Start the API server
python start_neuro_api.py

# Or directly
python api_server_neuro.py
```

The API will be available at `http://127.0.0.1:8000`

### API Endpoints

- `POST /optimize`: Optimize a prompt
  ```json
  {
    "raw_prompt": "Create a function",
    "max_length": 200,
    "num_return_sequences": 1
  }
  ```

- `GET /health`: Check server status

- `POST /feedback`: Submit user feedback (for future retraining)

## Key Advantages

1. **End-to-End Generation**: Actually generates improved prompt text (not just embeddings)
2. **Efficient Training**: Only trains small black box + projector (frozen encoder/generator)
3. **Soft Prompts**: Learns ghost tokens that enhance prompts without modifying base models
4. **Safety**: KL divergence penalty prevents gibberish
5. **Two-Phase Learning**: Distillation ensures validity, contrastive learning improves quality

## Differences from Previous Architecture

| Feature | Previous (VAE-style) | Neuro-Latent Optimizer |
|---------|---------------------|------------------------|
| Decoder | GPT-2 (not properly trained) | T5 (frozen, pre-trained) |
| Generation | Embedding → Text (problematic) | Soft Prompts → Text (working) |
| Training | Contrastive only | Distillation + Contrastive |
| Output | Embeddings (hard to decode) | Actual text generation |
| Model Size | Large (encoder + decoder trainable) | Small (only black box + projector) |

## Model Files

- `neuro_prompt_optimizer.py`: Core model implementation
- `train_neuro.py`: Training script
- `api_server_neuro.py`: FastAPI server
- `start_neuro_api.py`: Startup script

## Checkpoints

Checkpoints are saved as:
- `phase1_epoch_N.pt`: Phase 1 (Distillation) checkpoints
- `phase2_epoch_N.pt`: Phase 2 (Contrastive) checkpoints
- `neuro_final_model.pt`: Final trained model

The API server automatically loads the latest checkpoint on startup.


