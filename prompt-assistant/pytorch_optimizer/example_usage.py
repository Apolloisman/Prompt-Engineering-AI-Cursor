"""
Example usage of the Latent Prompt Optimizer
Demonstrates training, inference, and feedback integration
"""

import torch
from latent_prompt_optimizer import LatentPromptOptimizer, ContrastivePromptTrainer


def example_basic_usage():
    """Basic usage: optimize a single prompt"""
    print("=== Basic Usage Example ===")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = LatentPromptOptimizer(
        latent_dim=768,
        delta_hidden_dims=[1024, 512, 256],
        dropout_rate=0.2,
        device=device
    )
    
    # Optimize a prompt
    raw_prompt = "Create a website"
    print(f"\nRaw Prompt: {raw_prompt}")
    
    z, z_prime, _ = model.forward(raw_prompt, return_embeddings=False)
    print(f"Raw embedding shape: {z.shape}")
    print(f"Optimized embedding shape: {z_prime.shape}")
    
    # Compute similarity improvement (if we had a target)
    print("\n✓ Prompt encoded and optimized in latent space")


def example_training():
    """Example training step"""
    print("\n=== Training Example ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LatentPromptOptimizer(device=device)
    trainer = ContrastivePromptTrainer(model=model, learning_rate=1e-4)
    
    # Training triplets
    anchor_prompts = [
        "Create a website",
        "Fix the bug",
        "Explain how it works"
    ]
    
    positive_prompts = [
        "Create a modern, responsive website using HTML5, CSS3, and JavaScript. The website should be mobile-friendly and follow WCAG accessibility standards.",
        "Debug and fix the null pointer exception in the user authentication module. Add proper error handling and null checks.",
        "Explain how the authentication system works, including the token generation process, validation steps, and security measures."
    ]
    
    negative_prompts = [
        "website",
        "bug fix",
        "explain"
    ]
    
    # Training step
    metrics = trainer.train_step(anchor_prompts, positive_prompts, negative_prompts)
    
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Positive Similarity: {metrics['positive_similarity']:.4f}")
    print(f"Negative Similarity: {metrics['negative_similarity']:.4f}")
    print(f"Similarity Gap: {metrics['similarity_gap']:.4f}")
    print("\n✓ Training step completed")


def example_feedback_integration():
    """Example of updating model from user feedback"""
    print("\n=== Feedback Integration Example ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LatentPromptOptimizer(device=device)
    trainer = ContrastivePromptTrainer(model=model)
    
    # Simulate user editing a prompt
    raw_prompt = "Create a function"
    edited_prompt = "Create a Python function that takes two integers and returns their sum, with input validation and error handling."
    success_score = 0.9  # High success score
    
    print(f"Raw Prompt: {raw_prompt}")
    print(f"User Edited To: {edited_prompt}")
    print(f"Success Score: {success_score}")
    
    # Update model
    loss = trainer.update_from_feedback(raw_prompt, edited_prompt, success_score)
    
    print(f"Feedback Loss: {loss:.4f}")
    print("\n✓ Model updated from user feedback")


def example_rule_seeding():
    """Example of using rule embeddings"""
    print("\n=== Rule Seeding Example ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LatentPromptOptimizer(device=device)
    
    # Get rule embeddings
    rule_emb = model.get_rule_embeddings()
    print(f"Rule embedding shape: {rule_emb.shape}")
    print(f"Number of rules: {model.num_rules}")
    
    # Use specific rules
    specific_rules = torch.tensor([[0, 1, 2]]).to(device)  # Use rules 0, 1, 2
    specific_rule_emb = model.get_rule_embeddings(specific_rules)
    print(f"Specific rule embedding shape: {specific_rule_emb.shape}")
    
    print("\n✓ Rule embeddings retrieved")


def example_batch_processing():
    """Example of processing multiple prompts"""
    print("\n=== Batch Processing Example ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LatentPromptOptimizer(device=device)
    
    prompts = [
        "Create a website",
        "Fix the bug",
        "Explain how it works",
        "Optimize the code",
        "Create a function"
    ]
    
    print(f"Processing {len(prompts)} prompts...")
    
    optimized_embeddings = []
    for prompt in prompts:
        z, z_prime, _ = model.forward(prompt, return_embeddings=False)
        optimized_embeddings.append(z_prime)
    
    # Stack embeddings
    batch_embeddings = torch.cat(optimized_embeddings, dim=0)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    print("\n✓ Batch processing completed")


if __name__ == "__main__":
    print("Latent Prompt Optimizer - Example Usage\n")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_training()
        example_feedback_integration()
        example_rule_seeding()
        example_batch_processing()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()




