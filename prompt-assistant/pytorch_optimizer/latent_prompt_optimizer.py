"""
Black Box Prompt Optimization Engine
Latent Space Contrastive Learning for Prompt Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import RobertaModel, RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple, List, Optional, Dict
import numpy as np


class LatentPromptOptimizer(nn.Module):
    """
    VAE-Style Latent Space Optimizer for Prompts
    
    Architecture:
    1. Encoder: RoBERTa-large → Latent Embedding z
    2. Delta Network (Black Box): Deep Sequential → Optimized Embedding z'
    3. Decoder: GPT-2 → Reconstructed Text
    """
    
    def __init__(
        self,
        latent_dim: int = 768,
        delta_hidden_dims: List[int] = [1024, 512, 256],
        dropout_rate: float = 0.2,
        num_rules: int = 8,
        rule_embedding_dim: int = 128,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(LatentPromptOptimizer, self).__init__()
        
        self.latent_dim = latent_dim
        self.device = device
        
        # ========== ENCODER: Pre-trained RoBERTa ==========
        self.encoder_model = RobertaModel.from_pretrained('roberta-large')
        self.encoder_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.encoder_model.eval()  # Freeze encoder initially
        
        # Projection layer to match latent_dim
        encoder_output_dim = self.encoder_model.config.hidden_size  # 1024 for roberta-large
        self.encoder_projection = nn.Linear(encoder_output_dim, latent_dim)
        
        # ========== RULE SEEDING MECHANISM ==========
        # Fixed embedding layer for heuristic rules
        self.rule_embeddings = nn.Embedding(num_rules, rule_embedding_dim)
        self.num_rules = num_rules
        self.rule_embedding_dim = rule_embedding_dim
        
        # Initialize rule embeddings with meaningful values
        self._initialize_rule_embeddings()
        
        # Rule fusion layer
        self.rule_fusion = nn.Linear(rule_embedding_dim, latent_dim)
        
        # ========== DELTA NETWORK (Black Box) ==========
        # Deep Sequential block with 3+ layers to capture non-linear relationships
        delta_input_dim = latent_dim + latent_dim  # prompt + rule embeddings
        
        delta_layers = []
        prev_dim = delta_input_dim
        
        for hidden_dim in delta_hidden_dims:
            delta_layers.append(nn.Linear(prev_dim, hidden_dim))
            delta_layers.append(nn.ReLU())
            delta_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer: transforms to optimized embedding
        delta_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.delta_network = nn.Sequential(*delta_layers)
        
        # ========== DECODER: GPT-2 Generative Head ==========
        self.decoder_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.decoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        
        # Projection from latent space to decoder input
        decoder_input_dim = self.decoder_model.config.n_embd  # 768 for GPT-2
        self.decoder_projection = nn.Linear(latent_dim, decoder_input_dim)
        
        # ========== LOSS FUNCTIONS ==========
        self.cosine_similarity = CosineSimilarity(dim=1)
        
        self.to(device)
    
    def _initialize_rule_embeddings(self):
        """
        Initialize rule embeddings with semantic values
        Rules represent key prompt engineering principles
        """
        # Rule indices mapping:
        # 0: Be Specific
        # 1: Add Context
        # 2: Use Structure
        # 3: Include Examples
        # 4: Define Role
        # 5: Specify Format
        # 6: Add Constraints
        # 7: Include Verification
        
        # Initialize with small random values (will be learned)
        nn.init.normal_(self.rule_embeddings.weight, mean=0.0, std=0.1)
    
    def encode_prompt(self, prompt_text: str) -> torch.Tensor:
        """
        Encode raw prompt text into latent embedding z
        
        Args:
            prompt_text: Raw prompt string
            
        Returns:
            z: Latent embedding tensor [batch_size, latent_dim]
        """
        # Tokenize and encode with RoBERTa
        inputs = self.encoder_tokenizer(
            prompt_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder_model(**inputs)
            # Use [CLS] token embedding (pooler_output) or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                encoder_embedding = outputs.pooler_output
            else:
                # Mean pooling over sequence
                encoder_embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Project to latent dimension
        z = self.encoder_projection(encoder_embedding)
        
        return z
    
    def get_rule_embeddings(self, rule_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get rule embeddings for fusion
        
        Args:
            rule_indices: Optional tensor of rule indices to use
                         If None, uses all rules
        
        Returns:
            rule_emb: Fused rule embedding [batch_size, latent_dim]
        """
        if rule_indices is None:
            # Use all rules and average them
            all_rules = self.rule_embeddings.weight  # [num_rules, rule_embedding_dim]
            rule_emb = all_rules.mean(dim=0, keepdim=True)  # [1, rule_embedding_dim]
        else:
            # Use specified rules
            # rule_indices is 1D tensor like [0, 1, 2] for a single prompt
            rule_emb = self.rule_embeddings(rule_indices)  # [num_rules, rule_embedding_dim]
            
            # Average over rules to get single embedding for this prompt
            if rule_emb.dim() == 2:
                rule_emb = rule_emb.mean(dim=0, keepdim=True)  # [1, rule_embedding_dim]
            elif rule_emb.dim() == 3:
                rule_emb = rule_emb.mean(dim=1)  # Average over rules
        
        # Project to latent dimension
        rule_emb = self.rule_fusion(rule_emb)  # Should be [1, latent_dim] or [batch_size, latent_dim]
        
        # Ensure output is always 2D [batch_size, latent_dim]
        if rule_emb.dim() == 1:
            rule_emb = rule_emb.unsqueeze(0)  # [1, latent_dim]
        elif rule_emb.dim() == 3:
            # If somehow 3D, squeeze middle dimension
            rule_emb = rule_emb.squeeze(1) if rule_emb.size(1) == 1 else rule_emb.view(-1, rule_emb.size(-1))
        
        return rule_emb
    
    def apply_delta_network(self, z: torch.Tensor, rule_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply Black Box Delta Network to transform z → z'
        
        Args:
            z: Raw prompt embedding [batch_size, latent_dim]
            rule_emb: Rule embeddings [batch_size, latent_dim]
        
        Returns:
            z_prime: Optimized embedding [batch_size, latent_dim]
        """
        # Concatenate prompt embedding with rule embeddings
        combined = torch.cat([z, rule_emb], dim=1)  # [batch_size, 2*latent_dim]
        
        # Pass through Delta Network (Black Box)
        z_prime = self.delta_network(combined)  # [batch_size, latent_dim]
        
        return z_prime
    
    def decode_embedding(self, z_prime: torch.Tensor, max_length: int = 200) -> str:
        """
        Decode optimized embedding back to text
        
        Args:
            z_prime: Optimized embedding [batch_size, latent_dim]
            max_length: Maximum generation length
        
        Returns:
            Generated text string
        """
        # Project to decoder input dimension
        decoder_input = self.decoder_projection(z_prime)  # [batch_size, decoder_input_dim]
        
        # Expand to sequence length (for generation)
        # We'll use the embedding as initial hidden state
        batch_size = decoder_input.size(0)
        
        # Create initial input_ids (start token)
        start_token_id = self.decoder_tokenizer.bos_token_id or self.decoder_tokenizer.eos_token_id
        input_ids = torch.tensor([[start_token_id]] * batch_size).to(self.device)
        
        # Use decoder to generate
        with torch.no_grad():
            # For simplicity, we'll use the embedding to condition generation
            # In practice, you might want a more sophisticated approach
            outputs = self.decoder_model.generate(
                input_ids=input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.decoder_tokenizer.pad_token_id
            )
        
        # Decode to text
        generated_text = self.decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def forward(
        self,
        raw_prompt: str,
        rule_indices: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
        """
        Forward pass: Raw Prompt → Optimized Embedding → Text
        
        Args:
            raw_prompt: Raw prompt text
            rule_indices: Optional rule indices to use
            return_embeddings: If True, also return embeddings
        
        Returns:
            z: Raw embedding
            z_prime: Optimized embedding
            optimized_text: Optional decoded text
        """
        # Encode
        z = self.encode_prompt(raw_prompt)
        
        # Get rule embeddings
        rule_emb = self.get_rule_embeddings(rule_indices)
        
        # Apply Delta Network
        z_prime = self.apply_delta_network(z, rule_emb)
        
        # Decode (optional, expensive)
        optimized_text = None
        if return_embeddings:
            optimized_text = self.decode_embedding(z_prime)
        
        return z, z_prime, optimized_text


class ContrastivePromptTrainer:
    """
    Training loop for Contrastive Learning using Triplet Loss
    """
    
    def __init__(
        self,
        model: LatentPromptOptimizer,
        learning_rate: float = 1e-4,
        margin: float = 0.5,
        temperature: float = 0.07
    ):
        self.model = model
        self.margin = margin
        self.temperature = temperature
        
        # Optimizer (only train Delta Network and rule embeddings)
        trainable_params = list(model.delta_network.parameters()) + \
                          list(model.rule_embeddings.parameters()) + \
                          list(model.rule_fusion.parameters()) + \
                          list(model.encoder_projection.parameters()) + \
                          list(model.decoder_projection.parameters())
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        self.cosine_similarity = CosineSimilarity(dim=1)
    
    def compute_triplet_loss(
        self,
        anchor_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Triplet Loss: minimize distance to positive, maximize distance to negative
        
        Args:
            anchor_embedding: Output from Delta Network [batch_size, latent_dim]
            positive_embedding: Successful prompt embedding [batch_size, latent_dim]
            negative_embedding: Failed prompt embedding [batch_size, latent_dim]
        
        Returns:
            Loss tensor
        """
        # Cosine similarities
        pos_sim = self.cosine_similarity(anchor_embedding, positive_embedding)
        neg_sim = self.cosine_similarity(anchor_embedding, negative_embedding)
        
        # Triplet loss: maximize pos_sim, minimize neg_sim
        # Loss = max(0, margin - (pos_sim - neg_sim))
        loss = F.relu(self.margin - (pos_sim - neg_sim))
        
        return loss.mean()
    
    def compute_infonce_loss(
        self,
        anchor_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE Loss (Contrastive Learning)
        
        Args:
            anchor_embedding: Output from Delta Network [batch_size, latent_dim]
            positive_embedding: Successful prompt embedding [batch_size, latent_dim]
            negative_embeddings: Failed prompt embeddings [batch_size, num_negatives, latent_dim]
        
        Returns:
            Loss tensor
        """
        batch_size = anchor_embedding.size(0)
        
        # Compute similarities
        pos_sim = self.cosine_similarity(anchor_embedding, positive_embedding) / self.temperature
        
        # Compute similarities with negatives
        anchor_expanded = anchor_embedding.unsqueeze(1)  # [batch_size, 1, latent_dim]
        neg_sims = self.cosine_similarity(
            anchor_expanded.expand(-1, negative_embeddings.size(1), -1).contiguous().view(-1, self.model.latent_dim),
            negative_embeddings.view(-1, self.model.latent_dim)
        ) / self.temperature
        
        neg_sims = neg_sims.view(batch_size, -1)  # [batch_size, num_negatives]
        
        # Concatenate positive and negatives
        all_sims = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # [batch_size, 1 + num_negatives]
        
        # InfoNCE: -log(exp(pos_sim) / sum(exp(all_sims)))
        logits = all_sims
        labels = torch.zeros(batch_size, dtype=torch.long).to(self.model.device)  # Positive is at index 0
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def train_step(
        self,
        anchor_prompts: List[str],
        positive_prompts: List[str],
        negative_prompts: List[str],
        use_infonce: bool = False
    ) -> Dict[str, float]:
        """
        Single training step with triplet data
        
        Args:
            anchor_prompts: Raw prompts (input)
            positive_prompts: Successful prompts (target)
            negative_prompts: Failed prompts (negative)
            use_infonce: If True, use InfoNCE loss; else use Triplet loss
        
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Encode all prompts
        # encode_prompt returns [1, latent_dim], we need [batch_size, latent_dim]
        anchor_emb_list = []
        for prompt in anchor_prompts:
            emb = self.model.encode_prompt(prompt)  # [1, latent_dim]
            if emb.dim() == 3:
                emb = emb.squeeze(0)  # Remove batch dim if 3D
            elif emb.dim() == 1:
                emb = emb.unsqueeze(0)  # Add batch dim if 1D
            anchor_emb_list.append(emb)
        anchor_embeddings = torch.cat(anchor_emb_list, dim=0)  # [batch_size, latent_dim]
        
        positive_emb_list = []
        for prompt in positive_prompts:
            emb = self.model.encode_prompt(prompt)
            if emb.dim() == 3:
                emb = emb.squeeze(0)
            elif emb.dim() == 1:
                emb = emb.unsqueeze(0)
            positive_emb_list.append(emb)
        positive_embeddings = torch.cat(positive_emb_list, dim=0)
        
        negative_emb_list = []
        for prompt in negative_prompts:
            emb = self.model.encode_prompt(prompt)
            if emb.dim() == 3:
                emb = emb.squeeze(0)
            elif emb.dim() == 1:
                emb = emb.unsqueeze(0)
            negative_emb_list.append(emb)
        negative_embeddings = torch.cat(negative_emb_list, dim=0)
        
        # Get rule embeddings for each sample in batch
        batch_size = anchor_embeddings.size(0)
        rule_emb = self.model.get_rule_embeddings()  # Should be [1, latent_dim]
        
        # Ensure rule_emb is 2D [batch_size, latent_dim]
        # Handle any dimension issues
        while rule_emb.dim() > 2:
            rule_emb = rule_emb.squeeze(0)
        if rule_emb.dim() == 1:
            rule_emb = rule_emb.unsqueeze(0)
        
        # Expand to batch size - ensure we have [batch_size, latent_dim]
        if rule_emb.size(0) == 1:
            rule_emb = rule_emb.expand(batch_size, -1)
        elif rule_emb.size(0) != batch_size:
            # Take first and expand
            rule_emb = rule_emb[0:1].expand(batch_size, -1)
        
        # Verify shapes match
        assert anchor_embeddings.dim() == 2, f"anchor_embeddings should be 2D, got {anchor_embeddings.dim()}D"
        assert rule_emb.dim() == 2, f"rule_emb should be 2D, got {rule_emb.dim()}D"
        assert anchor_embeddings.size(0) == rule_emb.size(0), f"Batch sizes don't match: {anchor_embeddings.size(0)} vs {rule_emb.size(0)}"
        
        # Apply Delta Network to anchors
        anchor_optimized = self.model.apply_delta_network(anchor_embeddings, rule_emb)
        
        # Compute loss
        if use_infonce:
            # For InfoNCE, expand negatives
            negative_expanded = negative_embeddings.unsqueeze(1)  # [batch_size, 1, latent_dim]
            loss = self.compute_infonce_loss(anchor_optimized, positive_embeddings, negative_expanded)
        else:
            loss = self.compute_triplet_loss(anchor_optimized, positive_embeddings, negative_embeddings)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            pos_sim = self.cosine_similarity(anchor_optimized, positive_embeddings).mean().item()
            neg_sim = self.cosine_similarity(anchor_optimized, negative_embeddings).mean().item()
        
        return {
            'loss': loss.item(),
            'positive_similarity': pos_sim,
            'negative_similarity': neg_sim,
            'similarity_gap': pos_sim - neg_sim
        }
    
    def update_from_feedback(
        self,
        raw_prompt: str,
        edited_prompt: str,
        success_score: float
    ):
        """
        Update model weights based on user feedback
        
        Args:
            raw_prompt: Original prompt
            edited_prompt: User-edited (improved) prompt
            success_score: Success metric (0.0 to 1.0)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Encode both prompts
        raw_emb = self.model.encode_prompt(raw_prompt)
        edited_emb = self.model.encode_prompt(edited_prompt)
        
        # Get rule embeddings
        rule_emb = self.model.get_rule_embeddings()
        
        # Apply Delta Network to raw prompt
        raw_optimized = self.model.apply_delta_network(raw_emb, rule_emb)
        
        # Compute loss: move optimized embedding closer to edited embedding
        # Weighted by success score
        similarity = self.cosine_similarity(raw_optimized, edited_emb)
        loss = (1.0 - similarity) * success_score
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# Example usage and training script
if __name__ == "__main__":
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LatentPromptOptimizer(
        latent_dim=768,
        delta_hidden_dims=[1024, 512, 256],
        dropout_rate=0.2,
        device=device
    )
    
    # Initialize trainer
    trainer = ContrastivePromptTrainer(
        model=model,
        learning_rate=1e-4,
        margin=0.5
    )
    
    # Example training data
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
    print(f"Training Metrics: {metrics}")
    
    # Test forward pass
    raw_prompt = "Create a function"
    z, z_prime, optimized_text = model.forward(raw_prompt, return_embeddings=False)
    print(f"Raw embedding shape: {z.shape}")
    print(f"Optimized embedding shape: {z_prime.shape}")
    
    # Feedback update
    loss = trainer.update_from_feedback(
        raw_prompt="Create a function",
        edited_prompt="Create a Python function that takes two integers and returns their sum, with input validation and error handling.",
        success_score=0.9
    )
    print(f"Feedback loss: {loss}")


