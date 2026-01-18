"""
Saliency-based Explainer

Computes gradient-based saliency maps to identify important tokens.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class SaliencyExplainer:
    """
    Gradient-based saliency map computation.
    
    Computes |∂loss/∂input| to find tokens most influential for prediction.
    """
    
    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
    
    def compute_saliency(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_class: int = 1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute saliency scores for each token.
        
        Args:
            token_ids: [1, seq_len] - Input token IDs
            line_positions: [1, seq_len] - Line positions
            preceding_mask: [1, seq_len] - Preceding line mask
            attention_mask: [1, seq_len] - Padding mask
            target_class: Class to compute gradients for (1=defective)
        
        Returns:
            (saliency_scores, metadata)
        """
        self.model.eval()
        
        # Ensure inputs require gradients
        token_ids = token_ids.clone()
        
        # Get embeddings with gradient tracking
        if hasattr(self.model, 'level2') and hasattr(self.model.level2, 'token_embedding'):
            embeddings = self.model.level2.token_embedding(token_ids)
        elif hasattr(self.model, 'level2') and hasattr(self.model.level2, 'pretrained'):
            with torch.no_grad():
                embeddings = self.model.level2.pretrained(token_ids).last_hidden_state
                embeddings = self.model.level2.embedding_proj(embeddings)
        else:
            # Fallback: use one-hot encoding as proxy
            vocab_size = token_ids.max().item() + 1
            embeddings = torch.nn.functional.one_hot(token_ids, num_classes=vocab_size).float()
        
        embeddings.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(
            token_ids=token_ids,
            line_positions=line_positions,
            preceding_mask=preceding_mask,
            attention_mask=attention_mask
        )
        
        # Compute gradient w.r.t. target class
        if target_class == 1:
            target = outputs['defect_probability'].sum()
        else:
            target = (1 - outputs['defect_probability']).sum()
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Saliency = absolute gradient magnitude
        if embeddings.grad is not None:
            saliency = embeddings.grad.abs().sum(dim=-1)  # [1, seq_len]
            saliency = saliency.squeeze(0).detach().cpu().numpy()
        else:
            # Fallback
            saliency = np.zeros(token_ids.size(1))
        
        # Normalize
        if saliency.sum() > 0:
            saliency = saliency / saliency.sum()
        
        # Get top tokens
        top_indices = np.argsort(-saliency)[:10]
        
        metadata = {
            'method': 'gradient_saliency',
            'target_class': target_class,
            'prediction': outputs['defect_probability'].item(),
            'top_token_indices': top_indices.tolist(),
            'top_scores': saliency[top_indices].tolist()
        }
        
        return saliency, metadata
    
    def compute_integrated_gradients(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute Integrated Gradients for more accurate attribution.
        
        Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks"
        """
        self.model.eval()
        
        if baseline is None:
            # Use padding token as baseline
            baseline = torch.zeros_like(token_ids)
        
        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, n_steps)
        
        integrated_grads = torch.zeros_like(token_ids, dtype=torch.float)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (token_ids - baseline)
            interpolated = interpolated.long()
            
            # Compute gradients at this interpolation point
            saliency, _ = self.compute_saliency(
                interpolated, line_positions, preceding_mask, attention_mask
            )
            integrated_grads += torch.tensor(saliency)
        
        # Average and scale by (input - baseline)
        integrated_grads = integrated_grads / n_steps
        integrated_grads = integrated_grads.numpy()
        
        # Normalize
        if integrated_grads.sum() > 0:
            integrated_grads = integrated_grads / integrated_grads.sum()
        
        top_indices = np.argsort(-integrated_grads)[:10]
        
        metadata = {
            'method': 'integrated_gradients',
            'n_steps': n_steps,
            'top_token_indices': top_indices.tolist(),
            'top_scores': integrated_grads[top_indices].tolist()
        }
        
        return integrated_grads, metadata
    
    def get_top_tokens(
        self,
        saliency: np.ndarray,
        token_ids: torch.Tensor,
        k: int = 10
    ) -> List[Dict]:
        """Get top-k most important tokens with their strings."""
        top_indices = np.argsort(-saliency)[:k]
        
        tokens = []
        for idx in top_indices:
            token_info = {
                'position': int(idx),
                'score': float(saliency[idx]),
                'token_id': int(token_ids[0, idx].item())
            }
            
            if self.tokenizer is not None:
                token_info['token_string'] = self.tokenizer.decode([token_info['token_id']])
            
            tokens.append(token_info)
        
        return tokens
