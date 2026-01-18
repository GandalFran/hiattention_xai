"""
Level 5: Prediction & Fusion Head

Combines all hierarchical contexts (Level 2-4) and produces:
1. Defect probability (main prediction)
2. Token importance scores (explainability)
3. Confidence/uncertainty estimates

Uses cross-level attention for intelligent fusion of multi-level context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CrossLevelAttention(nn.Module):
    """
    Attention mechanism that fuses information across hierarchy levels.
    Allows each level to attend to other levels for context.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Multi-head attention for cross-level fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Level-specific projections
        self.level_projs = nn.ModuleDict({
            'local': nn.Linear(embedding_dim, embedding_dim),
            'function': nn.Linear(embedding_dim, embedding_dim),
            'architectural': nn.Linear(embedding_dim, embedding_dim)
        })
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        local_context: torch.Tensor,      # [batch, seq_len, E]
        function_context: torch.Tensor,   # [batch, E] or [batch, num_funcs, E]
        arch_context: torch.Tensor        # [batch, E] or [batch, num_modules, E]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse contexts from all levels.
        
        Returns:
            fused: [batch, embedding_dim] - Fused representation
            attention_weights: Attention patterns for explainability
        """
        batch_size = local_context.size(0)
        
        # Pool local context to get single vector
        local_pooled = local_context.mean(dim=1)  # [batch, E]
        
        # Ensure function and arch context are 2D
        if function_context.dim() == 3:
            function_pooled = function_context.mean(dim=1)
        else:
            function_pooled = function_context
        
        if arch_context.dim() == 3:
            arch_pooled = arch_context.mean(dim=1)
        else:
            arch_pooled = arch_context
        
        # Project each level
        local_proj = self.level_projs['local'](local_pooled)
        func_proj = self.level_projs['function'](function_pooled)
        arch_proj = self.level_projs['architectural'](arch_pooled)
        
        # Stack for attention: [batch, 3, E]
        stacked = torch.stack([local_proj, func_proj, arch_proj], dim=1)
        
        # Cross-level attention (each level attends to others)
        attended, attn_weights = self.attention(
            stacked, stacked, stacked,
            need_weights=True,
            average_attn_weights=False
        )
        
        # Flatten and project (use contiguous+reshape for attention output)
        fused = attended.contiguous().reshape(batch_size, -1)  # [batch, 3*E]
        fused = self.output_proj(fused)
        fused = self.layer_norm(fused)
        
        return fused, attn_weights


class DefectPredictionHead(nn.Module):
    """
    Binary classifier for defect prediction.
    Outputs probability of line being defective.
    """
    
    def __init__(self, embedding_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, embedding_dim]
        
        Returns:
            logits: [batch, 1] - Raw logits (apply sigmoid for probability)
        """
        return self.mlp(x)


class TokenImportanceHead(nn.Module):
    """
    Computes importance score for each token (for explainability).
    Helps identify which tokens contributed most to prediction.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        max_tokens: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        self.max_tokens = max_tokens
        
        # Attention-based importance scoring
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        # Final importance projection
        self.importance_proj = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        fused_context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute importance scores for each token.
        
        Args:
            token_embeddings: [batch, seq_len, embedding_dim]
            fused_context: [batch, embedding_dim] - Global context
            attention_mask: [batch, seq_len] - Padding mask
        
        Returns:
            importance_scores: [batch, seq_len] - Importance per token
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Use fused context as query
        query = self.query(fused_context).unsqueeze(1)  # [batch, 1, E]
        key = self.key(token_embeddings)                # [batch, seq_len, E]
        value = self.value(token_embeddings)            # [batch, seq_len, E]
        
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2))  # [batch, 1, seq_len]
        scores = scores / (token_embeddings.size(-1) ** 0.5)
        
        # Apply padding mask
        if attention_mask is not None:
            mask = ~attention_mask.bool()
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        # Softmax for attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Also compute per-token importance via projection
        importance_logits = self.importance_proj(token_embeddings).squeeze(-1)  # [batch, seq_len]
        
        # Combine attention-based and direct importance
        combined_importance = attention_weights.squeeze(1) * F.sigmoid(importance_logits)
        
        # Normalize
        if attention_mask is not None:
            combined_importance = combined_importance * attention_mask.float()
            sum_importance = combined_importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            combined_importance = combined_importance / sum_importance
        
        return combined_importance


class UncertaintyEstimationHead(nn.Module):
    """
    Estimates prediction uncertainty (epistemic and aleatoric).
    
    - Epistemic: Model uncertainty (can be reduced with more data)
    - Aleatoric: Data uncertainty (inherent noise)
    """
    
    def __init__(self, embedding_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [epistemic, aleatoric]
        )
        
        # Softplus to ensure positive variance
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, embedding_dim]
        
        Returns:
            epistemic: [batch] - Epistemic uncertainty
            aleatoric: [batch] - Aleatoric uncertainty
        """
        out = self.mlp(x)
        out = self.softplus(out)
        
        epistemic = out[:, 0]
        aleatoric = out[:, 1]
        
        return epistemic, aleatoric


class PredictionHead(nn.Module):
    """
    Complete Level 5 prediction head.
    
    Combines:
    1. Cross-level attention for fusion
    2. Defect probability prediction
    3. Token importance (explainability)
    4. Uncertainty estimation
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        max_tokens: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Cross-level fusion
        self.cross_level_attention = CrossLevelAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Prediction heads
        self.defect_head = DefectPredictionHead(embedding_dim, dropout)
        self.importance_head = TokenImportanceHead(embedding_dim, max_tokens, dropout)
        self.uncertainty_head = UncertaintyEstimationHead(embedding_dim, dropout)
    
    def forward(
        self,
        local_context: torch.Tensor,
        function_context: torch.Tensor,
        arch_context: torch.Tensor,
        token_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Combined forward pass through all prediction heads.
        
        Args:
            local_context: [batch, seq_len, E] - From Level 2
            function_context: [batch, E] - From Level 3
            arch_context: [batch, E] - From Level 4
            token_embeddings: [batch, seq_len, E] - Original token embeds
            attention_mask: [batch, seq_len] - Padding mask
        
        Returns:
            Dictionary with:
            - defect_logits: [batch, 1]
            - defect_probability: [batch, 1]
            - token_importance: [batch, seq_len]
            - epistemic_uncertainty: [batch]
            - aleatoric_uncertainty: [batch]
            - cross_level_attention: Attention weights
            - fused_embedding: [batch, E]
        """
        # Fuse all levels
        fused, cross_attn = self.cross_level_attention(
            local_context, function_context, arch_context
        )
        
        # Defect prediction
        defect_logits = self.defect_head(fused)
        defect_prob = torch.sigmoid(defect_logits)
        
        # Token importance (for explainability)
        if token_embeddings is not None:
            importance = self.importance_head(
                token_embeddings, fused, attention_mask
            )
        else:
            importance = self.importance_head(
                local_context, fused, attention_mask
            )
        
        # Uncertainty estimation
        epistemic, aleatoric = self.uncertainty_head(fused)
        
        return {
            'defect_logits': defect_logits,
            'defect_probability': defect_prob,
            'token_importance': importance,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'cross_level_attention': cross_attn,
            'fused_embedding': fused
        }
