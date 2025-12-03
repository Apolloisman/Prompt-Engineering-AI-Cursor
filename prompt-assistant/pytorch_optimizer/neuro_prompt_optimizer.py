"""
Neuro-Latent Optimizer: Soft Prompting + Latent Space Contrastive Learning
Uses Prefix Tuning (Soft Prompts) with T5 for generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from typing import Tuple, Optional, Dict
import numpy as np


class NeuroPromptOptimizer(nn.Module):
    """
    Neuro-Latent Optimizer using Soft Prompting (Prefix Tuning)
    
    Architecture:
    1. Encoder: RoBERTa-base (Frozen) → Raw Embedding
    2. Black Box: 3-layer Dense Network (Trainable) → Optimized Intent
    3. Projector: Optimized Intent → Soft Prompts (Ghost Tokens)
    4. Generator: T5-small (Frozen) → Enhanced Prompt Text
    """
    
    def __init__(
        self,
        encoder_model: str = 'roberta-base',
        generator_model: str = 't5-small',
        latent_dim: int = 768,
        soft_prompt_length: int = 20,
        generator_embed_dim: int = 512,
        black_box_hidden: int = 1024,
        dropout_rate: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(NeuroPromptOptimizer, self).__init__()
        
        self.latent_dim = latent_dim
        self.soft_prompt_length = soft_prompt_length
        self.generator_embed_dim = generator_embed_dim
        self.device = device
        
        # ========== ENCODER: RoBERTa-base (Frozen) ==========
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        encoder_output_dim = self.encoder.config.hidden_size  # 768 for roberta-base
        
        # ========== BLACK BOX: Trainable Core ==========
        # 3-layer dense network: 768 -> 1024 -> 768
        self.black_box = nn.Sequential(
            nn.Linear(encoder_output_dim, black_box_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(black_box_hidden, latent_dim)
        )
        
        # ========== PROJECTOR: Create Soft Prompts ==========
        # Projects optimized intent to soft prompt tokens
        # Output: [Batch, soft_prompt_length, generator_embed_dim]
        self.projector = nn.Linear(
            latent_dim,
            soft_prompt_length * generator_embed_dim
        )
        
        # ========== GENERATOR: T5-small (Trainable for fine-tuning) ==========
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model)
        # Generator will be unfrozen during training to learn prompt generation
        # Initially frozen, will be unfrozen by trainer
        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.train()  # Set to train mode but params frozen initially
        
        # Verify generator embedding dimension matches projector output
        actual_embed_dim = self.generator.config.d_model  # 512 for t5-small
        if actual_embed_dim != generator_embed_dim:
            print(f"Warning: Generator embed dim ({actual_embed_dim}) != specified ({generator_embed_dim})")
            # Adjust projector if needed
            self.projector = nn.Linear(
                latent_dim,
                soft_prompt_length * actual_embed_dim
            )
            self.generator_embed_dim = actual_embed_dim
    
    def encode_prompt(self, text: str) -> torch.Tensor:
        """
        Encode raw prompt text to embedding
        
        Args:
            text: Raw prompt string
            
        Returns:
            Raw embedding [1, latent_dim]
        """
        # Tokenize
        inputs = self.encoder_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Mean pooling over sequence length
            raw_embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
        
        return raw_embedding
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        raw_text: Optional[str] = None,
        max_length: int = 200,
        num_return_sequences: int = 1,
        return_soft_prompts: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Raw Prompt → Optimized Intent → Soft Prompts → Enhanced Text
        
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len] (optional)
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            raw_text: Raw prompt string (optional, used if input_ids not provided)
            max_length: Maximum generation length
            num_return_sequences: Number of sequences to generate
            return_soft_prompts: Whether to return soft prompt embeddings
            
        Returns:
            Dictionary with:
                - 'raw_embedding': Raw prompt embedding
                - 'optimized_intent': Optimized intent vector (after black box)
                - 'soft_prompts': Soft prompt tokens [batch, soft_prompt_length, embed_dim]
                - 'generated_ids': Generated token IDs
                - 'generated_text': Decoded generated text
        """
        # 1. Encode
        if raw_text is not None:
            raw_embedding = self.encode_prompt(raw_text)
            # Tokenize for generator
            gen_inputs = self.generator_tokenizer(
                raw_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            input_ids = gen_inputs['input_ids']
            attention_mask = gen_inputs['attention_mask']
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or raw_text must be provided")
            # Encode using encoder
            with torch.no_grad():
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                raw_embedding = encoder_outputs.last_hidden_state.mean(dim=1)
        
        batch_size = raw_embedding.size(0)
        
        # 2. Black Box "Thinking" (The Interrelated Connections)
        optimized_intent = self.black_box(raw_embedding)  # [batch_size, latent_dim]
        
        # 3. Create Ghost Tokens (Soft Prompts)
        # Project optimized intent to soft prompt space
        soft_prompt_flat = self.projector(optimized_intent)  # [batch_size, soft_prompt_length * embed_dim]
        soft_prompts = soft_prompt_flat.view(
            batch_size,
            self.soft_prompt_length,
            self.generator_embed_dim
        )  # [batch_size, soft_prompt_length, embed_dim]
        
        # 4. Concatenate with original input embeddings
        # Get input embeddings from generator
        input_embeds = self.generator.get_input_embeddings()(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Concatenate soft prompts with input embeddings
        combined_embeds = torch.cat([soft_prompts, input_embeds], dim=1)  # [batch_size, soft_prompt_length + seq_len, embed_dim]
        
        # Create attention mask for combined sequence
        soft_prompt_mask = torch.ones(
            batch_size,
            self.soft_prompt_length,
            device=self.device,
            dtype=attention_mask.dtype
        )
        combined_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)
        
        # 5. Generate
        with torch.no_grad():
            try:
                outputs = self.generator.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.generator_tokenizer.pad_token_id if self.generator_tokenizer.pad_token_id is not None else self.generator_tokenizer.eos_token_id,
                    eos_token_id=self.generator_tokenizer.eos_token_id
                )
            except Exception as e:
                # Fallback: if generation fails, return original text
                print(f"Generation failed: {e}")
                outputs = input_ids  # Return input as fallback
        
        # Decode generated text
        generated_texts = []
        for seq in outputs:
            try:
                decoded = self.generator_tokenizer.decode(seq, skip_special_tokens=True)
                generated_texts.append(decoded)
            except Exception as e:
                print(f"Decoding failed: {e}")
                # Fallback to original text
                if raw_text is not None:
                    generated_texts.append(raw_text)
                else:
                    generated_texts.append("")
        
        result = {
            'raw_embedding': raw_embedding,
            'optimized_intent': optimized_intent,
            'generated_ids': outputs,
            'generated_text': generated_texts[0] if len(generated_texts) == 1 else generated_texts
        }
        
        if return_soft_prompts:
            result['soft_prompts'] = soft_prompts
        
        return result
    
    def get_optimized_intent(self, raw_text: str) -> torch.Tensor:
        """
        Get optimized intent vector without generating text
        Useful for contrastive learning
        
        Args:
            raw_text: Raw prompt string
            
        Returns:
            Optimized intent vector [1, latent_dim]
        """
        raw_embedding = self.encode_prompt(raw_text)
        optimized_intent = self.black_box(raw_embedding)
        return optimized_intent


class ContrastiveNeuroTrainer:
    """
    Trainer for Neuro-Latent Optimizer with:
    - Phase 1: Distillation Loss (CrossEntropy)
    - Phase 2: Contrastive Loss (InfoNCE)
    - KL Divergence Penalty (Safety)
    """
    
    def __init__(
        self,
        model: NeuroPromptOptimizer,
        learning_rate: float = 1e-4,
        temperature: float = 0.07,
        kl_weight: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.device = device
        
        # Train black_box, projector, AND generator (unfreeze generator for fine-tuning)
        # Unfreeze generator for fine-tuning
        for param in model.generator.parameters():
            param.requires_grad = True
        
        trainable_params = list(model.black_box.parameters()) + \
                          list(model.projector.parameters()) + \
                          list(model.generator.parameters())
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
    def distillation_loss(
        self,
        generated_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Phase 1: Distillation Loss
        CrossEntropy between model output and teacher output
        
        Args:
            generated_logits: Model output logits [batch_size, vocab_size]
            teacher_logits: Teacher/target logits [batch_size, vocab_size]
            
        Returns:
            Distillation loss
        """
        # Softmax both
        generated_probs = F.log_softmax(generated_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL Divergence (CrossEntropy)
        loss = F.kl_div(generated_probs, teacher_probs, reduction='batchmean')
        return loss
    
    def contrastive_loss(
        self,
        anchor_intent: torch.Tensor,
        positive_intent: torch.Tensor,
        negative_intents: torch.Tensor
    ) -> torch.Tensor:
        """
        Phase 2: Contrastive Loss (InfoNCE)
        Pull anchor closer to positive, push away from negatives
        
        Args:
            anchor_intent: Optimized intent from raw prompt [batch_size, latent_dim]
            positive_intent: Intent from successful prompt [batch_size, latent_dim]
            negative_intents: Intents from failed prompts [batch_size, num_negatives, latent_dim] or [batch_size, latent_dim]
            
        Returns:
            InfoNCE loss
        """
        batch_size = anchor_intent.size(0)
        
        # Compute similarities
        # Normalize vectors
        anchor_norm = F.normalize(anchor_intent, p=2, dim=1)  # [batch_size, latent_dim]
        positive_norm = F.normalize(positive_intent, p=2, dim=1)  # [batch_size, latent_dim]
        
        # Positive similarity: [batch_size]
        pos_sim = (anchor_norm * positive_norm).sum(dim=1) / self.temperature  # [batch_size]
        
        # Handle negative intents - ensure it's 3D
        if negative_intents.dim() == 2:
            # [batch_size, latent_dim] -> [batch_size, 1, latent_dim]
            negative_intents = negative_intents.unsqueeze(1)
        elif negative_intents.dim() == 1:
            # [latent_dim] -> [1, 1, latent_dim] then expand
            negative_intents = negative_intents.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Normalize negative intents: [batch_size, num_negatives, latent_dim]
        negative_norm = F.normalize(negative_intents, p=2, dim=2)
        
        # Negative similarities: [batch_size, num_negatives]
        # Use bmm for batch matrix multiplication
        anchor_expanded = anchor_norm.unsqueeze(1)  # [batch_size, 1, latent_dim]
        negative_transposed = negative_norm.transpose(1, 2)  # [batch_size, latent_dim, num_negatives]
        
        neg_sims = torch.bmm(anchor_expanded, negative_transposed).squeeze(1) / self.temperature  # [batch_size, num_negatives]
        
        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # [batch_size, 1 + num_negatives]
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # Positive is at index 0
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def kl_penalty(
        self,
        soft_prompts: torch.Tensor,
        prior_mean: float = 0.0,
        prior_std: float = 1.0
    ) -> torch.Tensor:
        """
        KL Divergence Penalty for Soft Prompts
        Keeps soft prompts within readable distribution (prevents gibberish)
        
        Args:
            soft_prompts: Soft prompt embeddings [batch_size, soft_prompt_length, embed_dim]
            prior_mean: Prior mean (default: 0)
            prior_std: Prior std (default: 1)
            
        Returns:
            KL divergence penalty
        """
        # Flatten soft prompts
        flat_prompts = soft_prompts.view(-1, soft_prompts.size(-1))  # [batch * length, embed_dim]
        
        # Compute mean and std
        mean = flat_prompts.mean(dim=0)
        std = flat_prompts.std(dim=0) + 1e-8  # Add epsilon for stability
        
        # KL divergence: KL(N(mean, std) || N(prior_mean, prior_std))
        kl = 0.5 * (
            (std / prior_std).pow(2) +
            ((mean - prior_mean) / prior_std).pow(2) -
            1 +
            2 * torch.log(prior_std / std)
        ).sum()
        
        return kl
    
    def train_step_distillation(
        self,
        raw_prompts: list,
        teacher_prompts: list,
        return_loss_components: bool = False
    ) -> Dict[str, float]:
        """
        Phase 1 Training Step: Distillation with Text Generation Loss
        
        Args:
            raw_prompts: List of raw prompt strings
            teacher_prompts: List of teacher/enhanced prompt strings
            return_loss_components: Whether to return individual loss components
            
        Returns:
            Dictionary with loss metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = len(raw_prompts)
        total_gen_loss = 0.0
        
        # Process each prompt in batch
        for i, (raw_text, teacher_text) in enumerate(zip(raw_prompts, teacher_prompts)):
            # Get soft prompts and input embeddings
            raw_emb = self.model.encode_prompt(raw_text)
            opt_intent = self.model.black_box(raw_emb)
            soft_prompts_flat = self.model.projector(opt_intent)
            soft_prompts = soft_prompts_flat.view(1, self.model.soft_prompt_length, self.model.generator_embed_dim)
            
            # Tokenize inputs
            raw_inputs = self.model.generator_tokenizer(
                raw_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.model.device)
            
            teacher_inputs = self.model.generator_tokenizer(
                teacher_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.model.device)
            
            # Get input embeddings
            input_embeds = self.model.generator.get_input_embeddings()(raw_inputs['input_ids'])
            
            # Concatenate soft prompts with input
            combined_embeds = torch.cat([soft_prompts, input_embeds], dim=1)
            
            # Create attention mask
            soft_mask = torch.ones(1, self.model.soft_prompt_length, device=self.model.device, dtype=raw_inputs['attention_mask'].dtype)
            combined_mask = torch.cat([soft_mask, raw_inputs['attention_mask']], dim=1)
            
            # Forward pass through generator with teacher labels
            # T5 needs decoder_input_ids for training
            decoder_input_ids = self.model.generator_tokenizer(
                teacher_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=False
            ).to(self.model.device)['input_ids']
            
            # Create labels (shift decoder_input_ids)
            labels = decoder_input_ids.clone()
            labels[labels == self.model.generator_tokenizer.pad_token_id] = -100
            
            # T5 forward with encoder inputs (soft prompts + raw) and decoder inputs (teacher)
            outputs = self.model.generator(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            total_gen_loss += outputs.loss
        
        # Average generation loss
        gen_loss = total_gen_loss / batch_size
        
        # KL penalty on soft prompts (get from one sample)
        sample_emb = self.model.encode_prompt(raw_prompts[0])
        sample_intent = self.model.black_box(sample_emb)
        sample_soft_flat = self.model.projector(sample_intent)
        sample_soft = sample_soft_flat.view(1, self.model.soft_prompt_length, self.model.generator_embed_dim)
        kl_penalty_loss = self.kl_penalty(sample_soft)
        
        # Total loss: generation loss + KL penalty
        total_loss = gen_loss + self.kl_weight * kl_penalty_loss
        
        # Backward
        total_loss.backward()
        self.optimizer.step()
        
        metrics = {
            'loss': total_loss.item(),
            'generation_loss': gen_loss.item(),
            'kl_penalty': kl_penalty_loss.item()
        }
        
        return metrics
    
    def train_step_contrastive(
        self,
        anchor_prompts: list,
        positive_prompts: list,
        negative_prompts: list,
        return_loss_components: bool = False
    ) -> Dict[str, float]:
        """
        Phase 2 Training Step: Contrastive Learning
        
        Args:
            anchor_prompts: List of raw prompt strings
            positive_prompts: List of successful prompt strings
            negative_prompts: List of failed prompt strings (can be multiple per anchor)
            return_loss_components: Whether to return individual loss components
            
        Returns:
            Dictionary with loss metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get optimized intents
        anchor_intents = []
        for anchor_text in anchor_prompts:
            intent = self.model.get_optimized_intent(anchor_text)
            anchor_intents.append(intent)
        anchor_intents = torch.cat(anchor_intents, dim=0)  # [batch_size, latent_dim]
        
        # Get positive intents
        positive_intents = []
        for pos_text in positive_prompts:
            intent = self.model.get_optimized_intent(pos_text)
            positive_intents.append(intent)
        positive_intents = torch.cat(positive_intents, dim=0)  # [batch_size, latent_dim]
        
        # Get negative intents
        # Handle multiple negatives per anchor
        negative_intents_list = []
        for neg_text in negative_prompts:
            intent = self.model.get_optimized_intent(neg_text)  # [1, latent_dim]
            negative_intents_list.append(intent.squeeze(0))  # [latent_dim]
        
        # Stack all negatives: [num_negatives, latent_dim]
        negative_intents_flat = torch.stack(negative_intents_list, dim=0)
        
        # Reshape based on batch size
        batch_size = len(anchor_prompts)
        num_negatives = len(negative_prompts)
        num_negatives_per_anchor = num_negatives // batch_size
        
        if num_negatives_per_anchor == 1:
            # One negative per anchor: [batch_size, latent_dim] -> [batch_size, 1, latent_dim]
            negative_intents = negative_intents_flat.view(batch_size, 1, -1)
        else:
            # Multiple negatives per anchor: reshape to [batch_size, num_negatives_per_anchor, latent_dim]
            negative_intents = negative_intents_flat.view(batch_size, num_negatives_per_anchor, -1)
        
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(
            anchor_intents,
            positive_intents,
            negative_intents
        )
        
        # KL penalty (get soft prompts from one forward pass)
        sample_result = self.model(raw_text=anchor_prompts[0], return_soft_prompts=True)
        kl_penalty_loss = self.kl_penalty(sample_result['soft_prompts'])
        
        # Total loss
        total_loss = contrastive_loss + self.kl_weight * kl_penalty_loss
        
        # Backward
        total_loss.backward()
        self.optimizer.step()
        
        metrics = {
            'loss': total_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'kl_penalty': kl_penalty_loss.item()
        }
        
        return metrics

