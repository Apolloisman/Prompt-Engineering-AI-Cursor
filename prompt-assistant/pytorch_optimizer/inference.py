"""
Pure ML Inference: Load Flan-T5-base + LoRA adapters and generate optimized prompts
100% Neural Network - No rule-based fallbacks
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import sys
import argparse

sys.stdout.reconfigure(encoding='utf-8')

class PureMLPromptOptimizer:
    """
    Pure ML Prompt Optimizer using Flan-T5-base + LoRA
    No rule-based logic - 100% neural network
    """
    
    def __init__(self, base_model_name: str = "google/flan-t5-base", adapter_path: str = None, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        print(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        # Load LoRA adapters if provided
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading LoRA adapters from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print("✓ LoRA adapters loaded")
        else:
            print("⚠ No LoRA adapters found - using base model only")
        
        self.model.to(device)
        self.model.eval()
        print(f"Model ready on {device}")
    
    def optimize(self, vague_prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Pure ML optimization: Raw Input -> Tokenizer -> Flan-T5 (with LoRA) -> Optimized Output
        
        No if/else logic, no rule-based fallbacks - 100% neural network
        """
        # Tokenize input
        inputs = self.tokenizer(
            vague_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Generate using pure ML
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output (pure ML result)
        optimized_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return optimized_prompt

def main():
    parser = argparse.ArgumentParser(description='Pure ML Prompt Optimization Inference')
    parser.add_argument('--adapter_path', type=str, default='./checkpoints/lora_model/final_adapter',
                       help='Path to LoRA adapter directory')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Vague prompt to optimize')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum generation length')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = PureMLPromptOptimizer(adapter_path=args.adapter_path)
    
    # Optimize prompt (pure ML)
    print(f"\nInput: {args.prompt}")
    print("\nGenerating optimized prompt (Pure ML)...")
    
    optimized = optimizer.optimize(args.prompt, max_length=args.max_length)
    
    print(f"\nOptimized (Pure ML Output):\n{optimized}")

if __name__ == '__main__':
    import os
    main()

