"""
Train LoRA adapters on Flan-T5-base for Pure ML Prompt Optimization
Uses PEFT (Parameter-Efficient Fine-Tuning) to prevent catastrophic forgetting
"""

import json
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import argparse

sys.stdout.reconfigure(encoding='utf-8')

class PromptDataset(Dataset):
    """Dataset for prompt optimization training pairs"""
    
    def __init__(self, data_path: str, tokenizer, max_input_length: int = 128, max_target_length: int = 512):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.pairs = data['training_pairs']
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        print(f"Loaded {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Tokenize input (vague prompt)
        input_encodings = self.tokenizer(
            pair['input'],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (ideal prompt)
        target_encodings = self.tokenizer(
            pair['target'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'labels': target_encodings['input_ids'].squeeze()
        }

def setup_lora_model(model_name: str = "google/flan-t5-base", device: str = None):
    """
    Setup Flan-T5-base with LoRA adapters
    Only LoRA adapters are trainable, base model is frozen
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading base model: {model_name}")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank (low-rank dimension)
        lora_alpha=32,  # LoRA alpha (scaling factor)
        target_modules=["q", "v"],  # Target attention modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM  # Sequence-to-sequence task
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

def train(
    data_path: str,
    output_dir: str = "./checkpoints/lora_model",
    num_train_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 8,
    device: str = None
):
    """Train LoRA adapters on Flan-T5-base"""
    
    # Setup model and tokenizer
    model, tokenizer = setup_lora_model(device=device)
    
    # Load dataset
    print(f"\nLoading dataset from {data_path}...")
    dataset = PromptDataset(data_path, tokenizer)
    
    # Data collator for seq2seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=50,  # Reduced for faster start
        logging_steps=5,  # More frequent logging
        save_steps=62,  # Save after each epoch (62 batches)
        save_total_limit=5,  # Keep last 5 checkpoints
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
        dataloader_pin_memory=True,
        report_to="none",  # Disable wandb/tensorboard
        save_strategy="steps",  # Save based on steps
        evaluation_strategy="no"  # No evaluation during training
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("\n" + "="*60)
    print("STARTING LORA TRAINING")
    print("="*60)
    print(f"Epochs: {num_train_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Training Samples: {len(dataset)}")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    final_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\n✓ Training complete!")
    print(f"✓ LoRA adapters saved to {final_path}")
    print(f"✓ Base model remains unchanged (no catastrophic forgetting)")

def main():
    parser = argparse.ArgumentParser(description='Train LoRA adapters for Prompt Optimization')
    parser.add_argument('--data_path', type=str, default='./data/lora_training_data.json',
                       help='Path to training data JSON file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/lora_model',
                       help='Directory to save LoRA adapters')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (LoRA needs higher rate)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu), auto-detect if not specified')
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == '__main__':
    main()

