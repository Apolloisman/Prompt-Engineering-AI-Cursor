"""
Training Script for Latent Prompt Optimizer
Handles data loading, training loop, and model checkpointing
"""

import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse

from latent_prompt_optimizer import LatentPromptOptimizer, ContrastivePromptTrainer


class PromptDataset:
    """
    Dataset loader for contrastive training triplets
    """
    
    def __init__(self, data_path: str):
        """
        Load training data from JSON file
        
        Expected format:
        {
            "triplets": [
                {
                    "anchor": "raw prompt text",
                    "positive": "successful prompt text",
                    "negative": "failed prompt text",
                    "metadata": {...}
                },
                ...
            ]
        }
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.triplets = data.get('triplets', [])
        print(f"Loaded {len(self.triplets)} training triplets")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return {
            'anchor': triplet['anchor'],
            'positive': triplet['positive'],
            'negative': triplet['negative'],
            'metadata': triplet.get('metadata', {})
        }
    
    def get_batch(self, indices: List[int]) -> Tuple[List[str], List[str], List[str]]:
        """Get batch of triplets"""
        anchors = [self.triplets[i]['anchor'] for i in indices]
        positives = [self.triplets[i]['positive'] for i in indices]
        negatives = [self.triplets[i]['negative'] for i in indices]
        return anchors, positives, negatives


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Create a sample dataset for testing
    """
    sample_triplets = []
    
    # Example patterns
    patterns = [
        {
            'anchor': 'Create a website',
            'positive': 'Create a modern, responsive website using HTML5, CSS3 (flexbox/grid), and JavaScript. The website should be mobile-friendly, follow WCAG accessibility standards, and implement SEO best practices.',
            'negative': 'website'
        },
        {
            'anchor': 'Fix the bug',
            'positive': 'Debug and fix the null pointer exception in the user authentication module. Add proper error handling, null checks, and logging. Ensure the fix doesn\'t break existing functionality.',
            'negative': 'bug fix'
        },
        {
            'anchor': 'Explain how it works',
            'positive': 'Explain how the authentication system works, including: (1) token generation process, (2) validation steps, (3) security measures, and (4) error handling. Provide code examples.',
            'negative': 'explain'
        },
        {
            'anchor': 'Create a function',
            'positive': 'Create a Python function that takes two integers as parameters and returns their sum. Include input validation, type checking, error handling for edge cases, and comprehensive docstring.',
            'negative': 'function'
        },
        {
            'anchor': 'Optimize the code',
            'positive': 'Refactor and optimize the following code for better performance: analyze time complexity, reduce redundant operations, improve memory usage, and maintain readability. Provide before/after comparison.',
            'negative': 'optimize'
        }
    ]
    
    # Generate variations
    for i in range(num_samples):
        pattern = patterns[i % len(patterns)]
        sample_triplets.append({
            'anchor': pattern['anchor'],
            'positive': pattern['positive'],
            'negative': pattern['negative'],
            'metadata': {
                'sample_id': i,
                'pattern_type': i % len(patterns)
            }
        })
    
    dataset = {
        'triplets': sample_triplets,
        'metadata': {
            'num_samples': num_samples,
            'description': 'Sample dataset for prompt optimization training'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created sample dataset with {num_samples} triplets at {output_path}")


def train(
    model: LatentPromptOptimizer,
    trainer: ContrastivePromptTrainer,
    dataset: PromptDataset,
    num_epochs: int = 10,
    batch_size: int = 8,
    save_dir: str = './checkpoints',
    use_infonce: bool = False
):
    """
    Main training loop
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_batches = len(dataset) // batch_size
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # Shuffle indices
        indices = torch.randperm(len(dataset)).tolist()
        
        epoch_losses = []
        epoch_pos_sims = []
        epoch_neg_sims = []
        
        # Training loop
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}"):
            # Get batch indices
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # Get batch data
            anchors, positives, negatives = dataset.get_batch(batch_indices)
            
            # Training step
            metrics = trainer.train_step(
                anchors,
                positives,
                negatives,
                use_infonce=use_infonce
            )
            
            epoch_losses.append(metrics['loss'])
            epoch_pos_sims.append(metrics['positive_similarity'])
            epoch_neg_sims.append(metrics['negative_similarity'])
        
        # Epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_pos_sim = sum(epoch_pos_sims) / len(epoch_pos_sims)
        avg_neg_sim = sum(epoch_neg_sims) / len(epoch_neg_sims)
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Positive Similarity: {avg_pos_sim:.4f}")
        print(f"Negative Similarity: {avg_neg_sim:.4f}")
        print(f"Similarity Gap: {avg_pos_sim - avg_neg_sim:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': avg_loss,
            'metrics': {
                'positive_similarity': avg_pos_sim,
                'negative_similarity': avg_neg_sim,
                'similarity_gap': avg_pos_sim - avg_neg_sim
            }
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Latent Prompt Optimizer')
    parser.add_argument('--data_path', type=str, default='./data/training_data.json',
                       help='Path to training data JSON file')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5,
                       help='Margin for triplet loss')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--use_infonce', action='store_true',
                       help='Use InfoNCE loss instead of triplet loss')
    parser.add_argument('--latent_dim', type=int, default=768,
                       help='Latent dimension')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu), auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Create sample dataset if requested
    if args.create_sample:
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        create_sample_dataset(args.data_path, num_samples=100)
        return
    
    # Device setup
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = LatentPromptOptimizer(
        latent_dim=args.latent_dim,
        delta_hidden_dims=[1024, 512, 256],
        dropout_rate=0.2,
        device=device
    )
    
    # Initialize trainer
    trainer = ContrastivePromptTrainer(
        model=model,
        learning_rate=args.learning_rate,
        margin=args.margin
    )
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = PromptDataset(args.data_path)
    
    # Train
    print("Starting training...")
    train(
        model=model,
        trainer=trainer,
        dataset=dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        use_infonce=args.use_infonce
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()




