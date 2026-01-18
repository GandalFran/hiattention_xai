"""
Loss Functions

Implements loss functions for defect prediction:
- Weighted Binary Cross-Entropy (for class imbalance)
- Focal Loss (for hard example mining)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with class weighting.
    
    Handles class imbalance by weighting positive (defective) samples higher.
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch] - Raw predictions (before sigmoid)
            targets: [batch] - Binary labels (0 or 1)
        
        Returns:
            loss: Scalar loss value
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    
    FL(p) = -α(1-p)^γ * log(p)
    
    Where:
    - α: Class balance weight
    - γ: Focusing parameter (higher = focus more on hard examples)
    - p: Model confidence for correct class
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch] - Raw predictions
            targets: [batch] - Binary labels
        
        Returns:
            loss: Focal loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Binary cross entropy (unreduced)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute focal weight
        # For positive samples: (1 - p)^gamma
        # For negative samples: p^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components.
    
    Combines:
    - Main BCE loss for defect prediction
    - Uncertainty regularization
    - Explanation consistency (optional)
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        consistency_weight: float = 0.05,
        pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.uncertainty_weight = uncertainty_weight
        self.consistency_weight = consistency_weight
        
        self.bce_loss = WeightedBCELoss(pos_weight)
    
    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            outputs: Model outputs dict with:
                - defect_logits
                - epistemic_uncertainty
                - aleatoric_uncertainty
                - token_importance
            targets: [batch] - Binary labels
        
        Returns:
            Combined loss
        """
        # Main prediction loss
        bce = self.bce_loss(outputs['defect_logits'].squeeze(-1), targets)
        
        total_loss = self.bce_weight * bce
        
        # Uncertainty regularization
        # Encourage model to be certain about correct predictions
        if 'epistemic_uncertainty' in outputs:
            epistemic = outputs['epistemic_uncertainty']
            aleatoric = outputs['aleatoric_uncertainty']
            
            # Lower uncertainty for correctly classified samples
            probs = torch.sigmoid(outputs['defect_logits'].squeeze(-1))
            correct = ((probs > 0.5) == targets).float()
            
            uncertainty_loss = (epistemic + aleatoric) * correct
            total_loss += self.uncertainty_weight * uncertainty_loss.mean()
        
        # Explanation consistency (optional)
        # Token importance should sum to ~1
        if 'token_importance' in outputs and self.consistency_weight > 0:
            importance = outputs['token_importance']
            importance_sum = importance.sum(dim=-1)
            consistency_loss = (importance_sum - 1.0).abs().mean()
            total_loss += self.consistency_weight * consistency_loss
        
        return total_loss
