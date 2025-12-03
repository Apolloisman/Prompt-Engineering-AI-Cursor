"""
Training Script for Neuro-Latent Optimizer
Supports Phase 1 (Distillation) and Phase 2 (Contrastive) training
"""

import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
import sys

from neuro_prompt_optimizer import NeuroPromptOptimizer, ContrastiveNeuroTrainer

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class PromptDataset:
    """
    Dataset loader for training data
    Supports both distillation and contrastive formats
    """
    
    def __init__(self, data_path: str):
        """
        Load training data from JSON file
        
        Expected format for distillation:
        {
            "distillation_pairs": [
                {
                    "raw": "raw prompt text",
                    "teacher": "enhanced prompt text"
                },
                ...
            ]
        }
        
        Expected format for contrastive:
        {
            "triplets": [
                {
                    "anchor": "raw prompt text",
                    "positive": "successful prompt text",
                    "negative": "failed prompt text" or ["negative1", "negative2", ...]
                },
                ...
            ]
        }
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.distillation_pairs = data.get('distillation_pairs', [])
        self.triplets = data.get('triplets', [])
        
        print(f"Loaded {len(self.distillation_pairs)} distillation pairs")
        print(f"Loaded {len(self.triplets)} contrastive triplets")
    
    def get_distillation_batch(self, indices: List[int]) -> Tuple[List[str], List[str]]:
        """Get batch of distillation pairs"""
        raw_prompts = [self.distillation_pairs[i]['raw'] for i in indices]
        teacher_prompts = [self.distillation_pairs[i]['teacher'] for i in indices]
        return raw_prompts, teacher_prompts
    
    def get_contrastive_batch(self, indices: List[int]) -> Tuple[List[str], List[str], List[str]]:
        """Get batch of contrastive triplets"""
        anchors = [self.triplets[i]['anchor'] for i in indices]
        positives = [self.triplets[i]['positive'] for i in indices]
        
        # Handle multiple negatives
        negatives = []
        for i in indices:
            neg = self.triplets[i]['negative']
            if isinstance(neg, list):
                negatives.extend(neg)  # Flatten multiple negatives
            else:
                negatives.append(neg)
        
        return anchors, positives, negatives


def create_sample_dataset(data_path: str, num_distillation: int = 50, num_triplets: int = 50):
    """
    Create sample training dataset with both distillation and contrastive examples
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Sample distillation pairs
    distillation_pairs = [
        {
            "raw": "Create a function",
            "teacher": "Create a Python function that takes two integers as parameters and returns their sum. Include input validation, type checking, error handling for edge cases (negative numbers, zero), and a comprehensive docstring following Google style."
        },
        {
            "raw": "Build a website",
            "teacher": "Build a responsive website using HTML5, CSS3 (with Flexbox/Grid), and vanilla JavaScript. The website should be mobile-first, accessible (WCAG 2.1 AA), SEO-optimized, and include progressive enhancement. Use semantic HTML and ensure cross-browser compatibility."
        },
        {
            "raw": "Analyze data",
            "teacher": "Perform a comprehensive data analysis on the provided dataset. Include exploratory data analysis (EDA) with visualizations, statistical summaries, identification of patterns and outliers, correlation analysis, and actionable insights. Use Python with pandas, numpy, matplotlib/seaborn, and provide clear documentation."
        },
        {
            "raw": "Automate workflow",
            "teacher": "Design and implement an automated workflow system that integrates with existing APIs and databases. Include error handling, logging, retry mechanisms, monitoring, and notification systems. Ensure idempotency and handle edge cases gracefully."
        },
        {
            "raw": "Explain concept",
            "teacher": "Provide a detailed explanation of the concept with clear definitions, real-world examples, step-by-step breakdown, common misconceptions, and practical applications. Use analogies where helpful and include visual aids or code examples if applicable."
        }
    ]
    
    # Expand distillation pairs
    expanded_distillation = []
    for i in range(num_distillation):
        base_pair = distillation_pairs[i % len(distillation_pairs)]
        expanded_distillation.append({
            "raw": base_pair["raw"] + f" (variant {i})",
            "teacher": base_pair["teacher"]
        })
    
    # Sample triplets
    triplets = [
        {
            "anchor": "Create a function",
            "positive": "Create a Python function that takes two integers as parameters and returns their sum. Include input validation, type checking, error handling for edge cases (negative numbers, zero), and a comprehensive docstring following Google style.",
            "negative": "function"
        },
        {
            "anchor": "Build a website",
            "positive": "Build a responsive website using HTML5, CSS3 (with Flexbox/Grid), and vanilla JavaScript. The website should be mobile-first, accessible (WCAG 2.1 AA), SEO-optimized, and include progressive enhancement.",
            "negative": "website"
        },
        {
            "anchor": "Analyze data",
            "positive": "Perform a comprehensive data analysis on the provided dataset. Include exploratory data analysis (EDA) with visualizations, statistical summaries, identification of patterns and outliers.",
            "negative": "data"
        },
        {
            "anchor": "Automate workflow",
            "positive": "Design and implement an automated workflow system that integrates with existing APIs and databases. Include error handling, logging, retry mechanisms, monitoring, and notification systems.",
            "negative": "workflow"
        }
    ]
    
    # Expand triplets
    expanded_triplets = []
    for i in range(num_triplets):
        base_triplet = triplets[i % len(triplets)]
        expanded_triplets.append({
            "anchor": base_triplet["anchor"] + f" (variant {i})",
            "positive": base_triplet["positive"],
            "negative": base_triplet["negative"] + f" (bad {i})"
        })
    
    # Combine into dataset
    dataset = {
        "distillation_pairs": expanded_distillation,
        "triplets": expanded_triplets
    }
    
    # Save
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample dataset with {len(expanded_distillation)} distillation pairs and {len(expanded_triplets)} triplets")
    print(f"Saved to {data_path}")


def train_phase1_distillation(
    model: NeuroPromptOptimizer,
    trainer: ContrastiveNeuroTrainer,
    dataset: PromptDataset,
    num_epochs: int,
    batch_size: int,
    save_dir: str
):
    """
    Phase 1: Distillation Training
    """
    print("\n" + "="*60)
    print("PHASE 1: DISTILLATION TRAINING")
    print("="*60)
    
    if len(dataset.distillation_pairs) == 0:
        print("No distillation pairs found, skipping Phase 1")
        return
    
    num_batches = (len(dataset.distillation_pairs) + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        epoch_losses = []
        epoch_generation_losses = []
        epoch_kl_penalties = []
        
        # Shuffle indices
        indices = torch.randperm(len(dataset.distillation_pairs)).tolist()
        
        for batch_idx in tqdm(range(num_batches), desc=f"Phase 1 Epoch {epoch + 1}"):
            # Get batch
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            raw_prompts, teacher_prompts = dataset.get_distillation_batch(batch_indices)
            
            # Training step
            metrics = trainer.train_step_distillation(raw_prompts, teacher_prompts)
            
            epoch_losses.append(metrics['loss'])
            epoch_generation_losses.append(metrics['generation_loss'])
            epoch_kl_penalties.append(metrics['kl_penalty'])
        
        # Epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_generation = sum(epoch_generation_losses) / len(epoch_generation_losses)
        avg_kl = sum(epoch_kl_penalties) / len(epoch_kl_penalties)
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Generation Loss: {avg_generation:.4f}")
        print(f"KL Penalty: {avg_kl:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'phase1_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'phase': 'distillation',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': avg_loss,
            'metrics': {
                'generation_loss': avg_generation,
                'kl_penalty': avg_kl
            }
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


def train_phase2_contrastive(
    model: NeuroPromptOptimizer,
    trainer: ContrastiveNeuroTrainer,
    dataset: PromptDataset,
    num_epochs: int,
    batch_size: int,
    save_dir: str
):
    """
    Phase 2: Contrastive Training
    """
    print("\n" + "="*60)
    print("PHASE 2: CONTRASTIVE TRAINING")
    print("="*60)
    
    if len(dataset.triplets) == 0:
        print("No contrastive triplets found, skipping Phase 2")
        return
    
    num_batches = (len(dataset.triplets) + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        epoch_losses = []
        epoch_contrastive_losses = []
        epoch_kl_penalties = []
        
        # Shuffle indices
        indices = torch.randperm(len(dataset.triplets)).tolist()
        
        for batch_idx in tqdm(range(num_batches), desc=f"Phase 2 Epoch {epoch + 1}"):
            # Get batch
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            anchors, positives, negatives = dataset.get_contrastive_batch(batch_indices)
            
            # Training step
            metrics = trainer.train_step_contrastive(anchors, positives, negatives)
            
            epoch_losses.append(metrics['loss'])
            epoch_contrastive_losses.append(metrics['contrastive_loss'])
            epoch_kl_penalties.append(metrics['kl_penalty'])
        
        # Epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_contrastive = sum(epoch_contrastive_losses) / len(epoch_contrastive_losses)
        avg_kl = sum(epoch_kl_penalties) / len(epoch_kl_penalties)
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Contrastive Loss: {avg_contrastive:.4f}")
        print(f"KL Penalty: {avg_kl:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'phase2_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'phase': 'contrastive',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': avg_loss,
            'metrics': {
                'contrastive_loss': avg_contrastive,
                'kl_penalty': avg_kl
            }
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Neuro-Latent Optimizer')
    parser.add_argument('--data_path', type=str, default='./data/neuro_training_data.json',
                       help='Path to training data JSON file')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset')
    parser.add_argument('--phase1_epochs', type=int, default=3,
                       help='Number of Phase 1 (Distillation) epochs')
    parser.add_argument('--phase2_epochs', type=int, default=3,
                       help='Number of Phase 2 (Contrastive) epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=0.01,
                       help='KL divergence penalty weight')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu), auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Create sample dataset if requested
    if args.create_sample:
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        create_sample_dataset(args.data_path, num_distillation=50, num_triplets=50)
        return
    
    # Device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing Neuro-Latent Optimizer...")
    model = NeuroPromptOptimizer(device=device)
    model.to(device)
    
    # Initialize trainer
    trainer = ContrastiveNeuroTrainer(
        model=model,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        device=device
    )
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = PromptDataset(args.data_path)
    
    # Try to load Phase 1 checkpoint if Phase 2 is requested but Phase 1 is not
    if args.phase1_epochs == 0 and args.phase2_epochs > 0:
        phase1_checkpoint = os.path.join(args.save_dir, 'phase1_epoch_3.pt')
        if os.path.exists(phase1_checkpoint):
            print(f"Loading Phase 1 checkpoint from {phase1_checkpoint}...")
            checkpoint = torch.load(phase1_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Phase 1 checkpoint loaded successfully")
            else:
                model.load_state_dict(checkpoint)
                print("Phase 1 checkpoint loaded (legacy format)")
    
    # Phase 1: Distillation
    if args.phase1_epochs > 0:
        train_phase1_distillation(
            model, trainer, dataset,
            args.phase1_epochs,
            args.batch_size,
            args.save_dir
        )
    
    # Phase 2: Contrastive
    if args.phase2_epochs > 0:
        train_phase2_contrastive(
            model, trainer, dataset,
            args.phase2_epochs,
            args.batch_size,
            args.save_dir
        )
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'neuro_final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, final_path)
    print(f"\nSaved final model to {final_path}")


if __name__ == '__main__':
    main()

