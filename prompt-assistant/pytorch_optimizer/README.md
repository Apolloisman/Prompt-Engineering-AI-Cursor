# Black Box Prompt Optimization Engine

A PyTorch-based Latent Space Contrastive Learning system for automatically optimizing prompts using neural networks.

## Architecture Overview

The system uses a VAE-style architecture with three main components:

1. **Encoder**: Pre-trained RoBERTa-large model that compresses prompts into latent embeddings
2. **Delta Network (Black Box)**: Deep sequential neural network (3+ layers) that learns complex transformations from raw to optimized prompts
3. **Decoder**: GPT-2 model that reconstructs optimized embeddings back into text

## Key Features

- **Contrastive Learning**: Uses triplet loss or InfoNCE loss to learn from successful vs. failed prompts
- **Rule Seeding**: Initializes with heuristic rules (specificity, context, structure, etc.) that guide early learning
- **Feedback Integration**: Updates weights when users edit prompts, learning from real-world improvements
- **Non-linear Relationships**: Deep Delta Network captures complex, interrelated prompt optimization patterns

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Create Sample Dataset

```bash
python train.py --create_sample --data_path ./data/training_data.json
```

### 2. Train the Model

```bash
python train.py \
    --data_path ./data/training_data.json \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --save_dir ./checkpoints
```

### 3. Use Trained Model

```python
from latent_prompt_optimizer import LatentPromptOptimizer

# Load model
model = LatentPromptOptimizer(device='cuda')
model.load_state_dict(torch.load('checkpoints/checkpoint_epoch_10.pt')['model_state_dict'])

# Optimize a prompt
raw_prompt = "Create a website"
z, z_prime, optimized_text = model.forward(raw_prompt, return_embeddings=True)
print(f"Optimized: {optimized_text}")
```

### 4. Update from User Feedback

```python
from latent_prompt_optimizer import ContrastivePromptTrainer

trainer = ContrastivePromptTrainer(model=model)

# When user edits a prompt
trainer.update_from_feedback(
    raw_prompt="Create a function",
    edited_prompt="Create a Python function with input validation and error handling",
    success_score=0.9  # 0.0 to 1.0
)
```

## Training Data Format

Training data should be a JSON file with the following structure:

```json
{
  "triplets": [
    {
      "anchor": "raw prompt text",
      "positive": "successful/optimized prompt text",
      "negative": "failed/poor prompt text",
      "metadata": {
        "sample_id": 0,
        "pattern_type": "code_generation"
      }
    }
  ]
}
```

## Model Components

### LatentPromptOptimizer

Main model class that implements the VAE-style architecture.

**Key Methods:**
- `encode_prompt(prompt_text)`: Encodes text → latent embedding
- `apply_delta_network(z, rule_emb)`: Transforms raw → optimized embedding
- `decode_embedding(z_prime)`: Decodes embedding → text
- `forward(raw_prompt)`: Complete forward pass

### ContrastivePromptTrainer

Handles training with contrastive learning.

**Key Methods:**
- `train_step(anchors, positives, negatives)`: Single training step
- `compute_triplet_loss()`: Triplet loss calculation
- `compute_infonce_loss()`: InfoNCE loss calculation
- `update_from_feedback()`: Update from user edits

## Rule Seeding

The model starts with 8 heuristic rules embedded:
1. Be Specific
2. Add Context
3. Use Structure
4. Include Examples
5. Define Role
6. Specify Format
7. Add Constraints
8. Include Verification

These rules guide early learning, but the Delta Network learns to prioritize its own patterns as training progresses.

## Loss Functions

### Triplet Loss
Minimizes distance to positive examples while maximizing distance to negative examples:
```
Loss = max(0, margin - (sim(anchor, positive) - sim(anchor, negative)))
```

### InfoNCE Loss
Contrastive learning loss that treats optimization as a classification problem:
```
Loss = -log(exp(sim(anchor, positive)) / sum(exp(sim(anchor, all))))
```

## Hyperparameters

- `latent_dim`: Dimension of latent space (default: 768)
- `delta_hidden_dims`: Hidden layer dimensions for Delta Network (default: [1024, 512, 256])
- `dropout_rate`: Dropout probability (default: 0.2)
- `learning_rate`: Learning rate for optimizer (default: 1e-4)
- `margin`: Margin for triplet loss (default: 0.5)
- `temperature`: Temperature for InfoNCE loss (default: 0.07)

## Requirements

- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU recommended (but CPU works)

## Example Training Output

```
=== Epoch 1/10 ===
Epoch 1: 100%|████████| 12/12 [00:45<00:00,  3.78s/it]
Average Loss: 0.3245
Positive Similarity: 0.7823
Negative Similarity: 0.2341
Similarity Gap: 0.5482
Saved checkpoint: ./checkpoints/checkpoint_epoch_1.pt
```

## Future Enhancements

- [ ] Support for custom rule embeddings
- [ ] Multi-task learning with different prompt types
- [ ] Attention mechanisms in Delta Network
- [ ] Reinforcement learning from user feedback
- [ ] Model distillation for faster inference




