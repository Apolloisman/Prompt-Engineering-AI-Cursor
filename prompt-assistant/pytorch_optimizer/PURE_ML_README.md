# Pure ML Prompt Optimizer (LoRA)

## Overview

This is a **100% Neural Network** implementation using **LoRA (Low-Rank Adaptation)** with **Flan-T5-base**. No rule-based fallbacks, no if/else logic - pure ML.

## Architecture

### Base Model: `google/flan-t5-base`
- **Why**: Instruction-tuned, understands English and logic out of the box
- **Status**: Frozen (not trained) - prevents catastrophic forgetting

### Learning Layer: LoRA (PEFT)
- **Config**: `r=16`, `lora_alpha=32`, `target_modules=["q", "v"]`
- **What it does**: Inserts small trainable neural networks into the frozen base model
- **Result**: Learns "interrelated connections" between vague and ideal prompts

### Training Data: 500 "Golden Rule" Examples
- **Structure**: Vague Prompt → Ideal Structured Prompt
- **Rules embedded in data**: Persona, Context, Constraints, Steps, Goal
- **No hard-coded rules**: All rules learned implicitly by the model

## Files

1. **`data_gen.py`**: Generates 500 synthetic training pairs with rule-seeding
2. **`train_lora.py`**: Trains LoRA adapters (only adapters, base model frozen)
3. **`inference.py`**: Pure ML inference (no fallbacks)
4. **`api_server_lora.py`**: FastAPI server for pure ML optimization
5. **`start_lora_api.py`**: Startup script for API server

## Usage

### 1. Generate Training Data
```bash
python data_gen.py
```
Creates `./data/lora_training_data.json` with 500 training pairs.

### 2. Train LoRA Adapters
```bash
python train_lora.py --data_path ./data/lora_training_data.json --num_epochs 5 --learning_rate 1e-3 --batch_size 8
```
- Trains only LoRA adapters (base model frozen)
- Saves to `./checkpoints/lora_model/final_adapter`
- Prevents catastrophic forgetting

### 3. Run Inference (Pure ML)
```bash
python inference.py --adapter_path ./checkpoints/lora_model/final_adapter --prompt "create function"
```

### 4. Start API Server
```bash
python start_lora_api.py
```
API available at `http://127.0.0.1:8000`

## Key Features

✅ **Pure ML**: 100% neural network, no rule-based code  
✅ **No Catastrophic Forgetting**: Base model frozen, only LoRA trained  
✅ **Rule-Learning**: Rules learned implicitly from training data  
✅ **High Quality**: Flan-T5-base is instruction-tuned  
✅ **Efficient**: LoRA uses <1% of base model parameters  

## Pipeline

```
Raw Input (Vague Prompt)
    ↓
Tokenizer
    ↓
Flan-T5-base (Frozen)
    ↓
LoRA Adapters (Trained)
    ↓
Optimized Output (Ideal Prompt)
```

**No if/else logic. No rule-based fallbacks. Pure ML.**

## Training Hyperparameters

- **Epochs**: 5 (LoRA converges fast)
- **Learning Rate**: 1e-3 (LoRA needs higher rate)
- **Batch Size**: 8 or 16
- **Base Model**: Frozen (not trained)
- **Trainable**: Only LoRA adapters (~0.1% of parameters)

## Expected Output Format

The model learns to generate structured prompts:

```
Act as a [persona].

Context: [why this is needed]

Task: [the vague prompt]

Constraints:
- [specific requirement 1]
- [specific requirement 2]
...

Steps:
1. [step 1]
2. [step 2]
...

Goal: [desired outcome]
```

This structure is learned from the 500 training examples, not hard-coded.


