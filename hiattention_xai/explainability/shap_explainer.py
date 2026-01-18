"""
SHAP-based Explainer

Computes SHAP (SHapley Additive exPlanations) values for token importance.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable
import warnings


class SHAPExplainer:
    """
    SHAP value computation for defect prediction explainability.
    
    Uses Kernel SHAP for model-agnostic explanations.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Try to import shap
        try:
            import shap
            self.shap_available = True
        except ImportError:
            self.shap_available = False
            warnings.warn("SHAP not installed. Using simplified approximation.")
    
    def compute_shap_values(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_samples: int = 100
    ) -> Dict:
        """
        Compute SHAP values for each token.
        
        Args:
            token_ids: [1, seq_len] - Token IDs
            line_positions: [1, seq_len] - Line positions
            preceding_mask: [1, seq_len] - Preceding mask
            attention_mask: [1, seq_len] - Attention mask
            n_samples: Number of samples for SHAP approximation
        
        Returns:
            Dictionary with SHAP values and metadata
        """
        self.model.eval()
        seq_len = token_ids.size(1)
        
        if self.shap_available:
            return self._compute_kernel_shap(
                token_ids, line_positions, preceding_mask, attention_mask, n_samples
            )
        else:
            return self._compute_approximate_shap(
                token_ids, line_positions, preceding_mask, attention_mask, n_samples
            )
    
    def _compute_kernel_shap(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        n_samples: int
    ) -> Dict:
        """Compute SHAP using the shap library."""
        import shap
        
        seq_len = token_ids.size(1)
        
        # Define prediction function for SHAP
        def predict_fn(mask_array):
            """Predict with masked tokens."""
            predictions = []
            
            for mask in mask_array:
                # Create masked input
                masked_ids = token_ids.clone()
                for i, m in enumerate(mask):
                    if m == 0:  # Token is masked
                        masked_ids[0, i] = self.tokenizer.pad_token_id or 0
                
                with torch.no_grad():
                    outputs = self.model(
                        token_ids=masked_ids.to(self.device),
                        line_positions=line_positions.to(self.device),
                        preceding_mask=preceding_mask.to(self.device),
                        attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
                    )
                    predictions.append(outputs['defect_probability'].item())
            
            return np.array(predictions)
        
        # Create background (all tokens masked)
        background = np.zeros((1, seq_len))
        
        # Create explainer
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Compute SHAP values
        test_input = np.ones((1, seq_len))
        shap_values = explainer.shap_values(test_input, nsamples=n_samples)
        
        # Extract values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values.flatten()
        
        # Get top contributors
        top_positive = np.argsort(-shap_values)[:5]
        top_negative = np.argsort(shap_values)[:5]
        
        return {
            'method': 'kernel_shap',
            'shap_values': shap_values.tolist(),
            'top_positive': [
                {'position': int(i), 'value': float(shap_values[i]),
                 'token': self.tokenizer.decode([token_ids[0, i].item()]) if self.tokenizer else str(i)}
                for i in top_positive
            ],
            'top_negative': [
                {'position': int(i), 'value': float(shap_values[i]),
                 'token': self.tokenizer.decode([token_ids[0, i].item()]) if self.tokenizer else str(i)}
                for i in top_negative
            ],
            'expected_value': float(predict_fn(background)[0])
        }
    
    def _compute_approximate_shap(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        n_samples: int
    ) -> Dict:
        """
        Approximate SHAP values using perturbation sampling.
        
        This is a simplified version when full SHAP is not available.
        """
        seq_len = token_ids.size(1)
        shap_values = np.zeros(seq_len)
        
        # Get baseline prediction (all tokens)
        with torch.no_grad():
            outputs = self.model(
                token_ids=token_ids.to(self.device),
                line_positions=line_positions.to(self.device),
                preceding_mask=preceding_mask.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
            )
            baseline_pred = outputs['defect_probability'].item()
        
        # Compute marginal contribution of each token
        for i in range(min(seq_len, 100)):  # Limit for efficiency
            contributions = []
            
            for _ in range(min(n_samples, 20)):
                # Random subset of tokens to include
                subset = np.random.choice([0, 1], size=seq_len, p=[0.5, 0.5])
                
                # Prediction without token i
                subset_without = subset.copy()
                subset_without[i] = 0
                
                # Prediction with token i
                subset_with = subset.copy()
                subset_with[i] = 1
                
                pred_without = self._predict_with_mask(
                    token_ids, line_positions, preceding_mask, attention_mask, subset_without
                )
                pred_with = self._predict_with_mask(
                    token_ids, line_positions, preceding_mask, attention_mask, subset_with
                )
                
                contributions.append(pred_with - pred_without)
            
            shap_values[i] = np.mean(contributions)
        
        # Normalize
        if np.abs(shap_values).sum() > 0:
            shap_values = shap_values / np.abs(shap_values).sum()
        
        top_positive = np.argsort(-shap_values)[:5]
        top_negative = np.argsort(shap_values)[:5]
        
        return {
            'method': 'approximate_shap',
            'shap_values': shap_values.tolist(),
            'top_positive': [
                {'position': int(i), 'value': float(shap_values[i])}
                for i in top_positive
            ],
            'top_negative': [
                {'position': int(i), 'value': float(shap_values[i])}
                for i in top_negative
            ],
            'expected_value': baseline_pred
        }
    
    def _predict_with_mask(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        mask: np.ndarray
    ) -> float:
        """Make prediction with masked tokens."""
        masked_ids = token_ids.clone()
        
        for i, m in enumerate(mask):
            if m == 0:
                masked_ids[0, i] = self.tokenizer.pad_token_id if self.tokenizer else 0
        
        with torch.no_grad():
            outputs = self.model(
                token_ids=masked_ids.to(self.device),
                line_positions=line_positions.to(self.device),
                preceding_mask=preceding_mask.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
            )
        
        return outputs['defect_probability'].item()
