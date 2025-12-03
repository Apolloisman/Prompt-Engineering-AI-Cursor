# Implementation Details

## Architecture Overview

The Black Box Prompt Optimization Engine implements a **Latent Space Contrastive Learning** approach using a VAE-style architecture.

### Component Breakdown

#### 1. Encoder (RoBERTa-large)
- **Purpose**: Compress raw prompt text into high-dimensional latent embeddings
- **Input**: Raw prompt string
- **Output**: Latent embedding `z` of dimension 768 (configurable)
- **Implementation**: 
  - Uses pre-trained `roberta-large` model
  - Projects 1024-dim RoBERTa output to 768-dim latent space
  - Uses [CLS] token or mean pooling

#### 2. Rule Seeding Mechanism
- **Purpose**: Initialize learning with heuristic rules
- **Implementation**:
  - Fixed embedding layer with 8 rules (128-dim each)
  - Rules: Specificity, Context, Structure, Examples, Role, Format, Constraints, Verification
  - Rule embeddings are concatenated with prompt embeddings
  - Fusion layer projects rules to latent dimension

#### 3. Delta Network (Black Box)
- **Purpose**: Learn complex transformations from raw → optimized prompts
- **Architecture**: Deep Sequential network
  - Input: Concatenated [prompt_embedding, rule_embedding] (1536-dim)
  - Hidden layers: [1024, 512, 256] (configurable)
  - Output: Optimized embedding `z'` (768-dim)
  - Activation: ReLU with Dropout (0.2)
- **Key Feature**: 3+ layers capture non-linear, interrelated relationships

#### 4. Decoder (GPT-2)
- **Purpose**: Reconstruct optimized embeddings back to text
- **Implementation**: 
  - Uses pre-trained GPT-2 model
  - Projects latent embedding to GPT-2 input dimension
  - Generates text from optimized embedding
  - Note: Currently basic implementation; can be enhanced

## Training Mechanism

### Contrastive Learning

The model learns from **triplet data**:
- **Anchor**: Raw prompt (input)
- **Positive**: Successful/optimized prompt (target)
- **Negative**: Failed/poor prompt (negative example)

### Loss Functions

#### 1. Triplet Loss
```python
pos_sim = cosine_similarity(anchor_optimized, positive)
neg_sim = cosine_similarity(anchor_optimized, negative)
loss = max(0, margin - (pos_sim - neg_sim))
```

**Objective**: 
- Minimize distance to positive examples
- Maximize distance to negative examples
- Maintain margin between them

#### 2. InfoNCE Loss (Alternative)
```python
# Treats as classification problem
logits = [sim(anchor, positive), sim(anchor, neg1), sim(anchor, neg2), ...]
loss = cross_entropy(logits, label=0)  # Positive is at index 0
```

**Objective**: 
- Learn to distinguish positive from negatives
- More robust with multiple negatives

### Training Process

1. **Encode all prompts** (anchor, positive, negative) → embeddings
2. **Apply Delta Network** to anchor → optimized embedding
3. **Compute loss** between optimized embedding and positive/negative
4. **Backpropagate** through Delta Network (encoder/decoder frozen initially)
5. **Update weights** to minimize loss

## Feedback Integration

When users edit prompts, the system learns from the delta:

```python
raw_emb = encode(raw_prompt)
edited_emb = encode(edited_prompt)
optimized_emb = delta_network(raw_emb)

loss = (1 - cosine_similarity(optimized_emb, edited_emb)) * success_score
```

**Key Points**:
- Success score (0.0-1.0) weights the update
- Model learns that specific edits improve prompts
- Continuous learning from real-world usage

## Mathematical Formulation

### Latent Space Transformation

Given:
- Raw prompt: `x_raw`
- Encoder: `E: text → z`
- Delta Network: `Δ: z → z'`
- Decoder: `D: z' → text`

The optimization process:
```
z = E(x_raw)                    # Encode to latent space
z' = Δ(z, rule_emb)            # Apply learned transformation
x_optimized = D(z')            # Decode back to text
```

### Learning Objective

For triplet `(anchor, positive, negative)`:
```
L = max(0, margin - (sim(Δ(E(anchor)), E(positive)) - sim(Δ(E(anchor)), E(negative))))
```

Where `sim` is cosine similarity.

## Key Design Decisions

### 1. Why VAE-Style Architecture?
- **Latent space** allows learning abstract prompt patterns
- **Decoder** enables reconstruction (though optional)
- **Bottleneck** forces learning of essential transformations

### 2. Why Deep Delta Network?
- **Non-linear relationships**: Prompt optimization involves complex, interrelated factors
- **3+ layers**: Necessary to capture these relationships
- **Dropout**: Prevents overfitting to training patterns

### 3. Why Rule Seeding?
- **Cold start problem**: Model needs initial guidance
- **Hybrid approach**: Combines rules with learned patterns
- **Gradual transition**: Model learns to prioritize learned patterns over rules

### 4. Why Contrastive Learning?
- **No explicit labels**: We don't have "goodness scores"
- **Relative learning**: Learn what makes prompts better/worse
- **Robust**: Works with diverse prompt types

## Performance Considerations

### Memory
- RoBERTa-large: ~355M parameters
- GPT-2: ~124M parameters
- Delta Network: ~2M parameters (trainable)
- **Total**: ~500M parameters (most frozen)

### Speed
- Encoding: ~50-100ms per prompt (GPU)
- Delta Network: ~1-2ms per prompt
- Decoding: ~100-200ms per prompt (generation)
- **Total**: ~200-300ms per optimization (GPU)

### Optimization Tips
1. **Freeze encoder/decoder**: Only train Delta Network
2. **Batch processing**: Process multiple prompts together
3. **Quantization**: Use INT8 for inference
4. **Model distillation**: Train smaller student model

## Future Enhancements

1. **Attention Mechanisms**: Add attention in Delta Network
2. **Multi-task Learning**: Different optimizers for different prompt types
3. **Reinforcement Learning**: Learn from user success metrics
4. **Few-shot Learning**: Adapt quickly to new prompt domains
5. **Interpretability**: Visualize what Delta Network learns

## Usage Patterns

### Pattern 1: Offline Training
```python
# Train on large dataset
trainer.train_step(anchors, positives, negatives)
# Save checkpoint
torch.save(model.state_dict(), 'checkpoint.pt')
```

### Pattern 2: Online Learning
```python
# Continuous learning from user feedback
for user_edit in user_edits:
    trainer.update_from_feedback(raw, edited, score)
```

### Pattern 3: Hybrid
```python
# Pre-train on dataset
train_on_dataset()
# Fine-tune on user feedback
for feedback in feedback_stream:
    update_from_feedback()
```

## Evaluation Metrics

1. **Similarity Gap**: `sim(optimized, positive) - sim(optimized, negative)`
   - Higher is better
   - Measures separation between good/bad prompts

2. **Positive Similarity**: `sim(optimized, positive)`
   - Target: > 0.8
   - Measures how close to successful prompts

3. **Negative Similarity**: `sim(optimized, negative)`
   - Target: < 0.3
   - Measures distance from failed prompts

4. **User Success Rate**: Percentage of optimized prompts that users accept
   - Real-world metric
   - Requires user feedback

## Troubleshooting

### Issue: Loss not decreasing
- **Check**: Learning rate (try 1e-5 or 1e-3)
- **Check**: Margin value (try 0.3 or 0.7)
- **Check**: Data quality (ensure clear positive/negative distinction)

### Issue: Overfitting
- **Solution**: Increase dropout (0.3-0.5)
- **Solution**: Add more training data
- **Solution**: Reduce Delta Network size

### Issue: Slow training
- **Solution**: Use smaller batch size
- **Solution**: Freeze encoder/decoder
- **Solution**: Use gradient accumulation

### Issue: Poor optimization quality
- **Solution**: Train longer (more epochs)
- **Solution**: Improve training data quality
- **Solution**: Adjust rule embeddings




