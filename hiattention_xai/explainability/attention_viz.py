"""
Attention Visualization

Visualizes attention patterns from the model for explainability.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class AttentionVisualizer:
    """
    Visualizes attention patterns from HiAttention-XAI model.
    
    Supports:
    - Self-attention heatmaps
    - Cross-level attention visualization
    - Head-specific pattern analysis
    """
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
    
    def extract_attention(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from model forward pass.
        
        Returns:
            Dictionary with attention matrices from different layers
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(
                token_ids=token_ids,
                line_positions=line_positions,
                preceding_mask=preceding_mask,
                attention_mask=attention_mask
            )
        
        attention_dict = {}
        
        # Local attention (Level 2)
        if 'local_attention' in outputs:
            local_attn = outputs['local_attention']
            if isinstance(local_attn, torch.Tensor):
                attention_dict['local_attention'] = local_attn.cpu().numpy()
        
        # Cross-level attention (Level 5)
        if 'cross_level_attention' in outputs:
            cross_attn = outputs['cross_level_attention']
            if isinstance(cross_attn, torch.Tensor):
                attention_dict['cross_level_attention'] = cross_attn.cpu().numpy()
        
        return attention_dict
    
    def plot_attention_heatmap(
        self,
        attention: np.ndarray,
        tokens: Optional[List[str]] = None,
        title: str = "Attention Pattern",
        head_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot attention weights as heatmap.
        
        Args:
            attention: [batch, heads, seq, seq] or [seq, seq]
            tokens: Token strings for axis labels
            title: Plot title
            head_idx: Which attention head to visualize
            save_path: Path to save figure
            figsize: Figure size
        """
        # Handle different attention shapes
        if attention.ndim == 4:
            attn = attention[0, head_idx]  # [seq, seq]
        elif attention.ndim == 3:
            attn = attention[head_idx]
        else:
            attn = attention
        
        # Limit size for visualization
        max_size = 50
        if attn.shape[0] > max_size:
            attn = attn[:max_size, :max_size]
            if tokens:
                tokens = tokens[:max_size]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            attn,
            ax=ax,
            cmap='viridis',
            square=True,
            xticklabels=tokens[:min(len(tokens), max_size)] if tokens else False,
            yticklabels=tokens[:min(len(tokens), max_size)] if tokens else False,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Key Positions', fontsize=12)
        ax.set_ylabel('Query Positions', fontsize=12)
        
        # Rotate tick labels
        if tokens:
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_attention_heads(
        self,
        attention: np.ndarray,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot multiple attention heads in a grid.
        """
        if attention.ndim < 3:
            print("Attention must have head dimension")
            return
        
        if attention.ndim == 4:
            attention = attention[0]  # Remove batch dim
        
        n_heads = min(attention.shape[0], 8)  # Limit to 8 heads
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(n_heads):
            ax = axes[i]
            attn = attention[i][:30, :30]  # Limit size
            
            sns.heatmap(
                attn,
                ax=ax,
                cmap='viridis',
                square=True,
                xticklabels=False,
                yticklabels=False,
                cbar=False
            )
            ax.set_title(f'Head {i}', fontsize=10)
        
        # Hide unused axes
        for i in range(n_heads, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Attention Head Patterns', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def classify_attention_pattern(
        self,
        attention: np.ndarray
    ) -> Dict[str, any]:
        """
        Analyze and classify attention pattern type.
        
        Patterns:
        - Sequential: Attends to nearby positions
        - Variable-focused: Attends to specific positions
        - Broad: Uniform attention distribution
        """
        if attention.ndim > 2:
            attention = attention.mean(axis=tuple(range(attention.ndim - 2)))
        
        seq_len = attention.shape[0]
        
        # Diagonal score (sequential pattern)
        diag_sum = np.trace(attention) / seq_len
        
        # Entropy (broad vs focused)
        # Higher entropy = more uniform
        flat = attention.flatten()
        flat = flat[flat > 0]  # Remove zeros
        if len(flat) > 0:
            entropy = -np.sum(flat * np.log(flat + 1e-10)) / np.log(len(flat))
        else:
            entropy = 0
        
        # Max focus (variable-focused pattern)
        max_per_row = attention.max(axis=1)
        max_focus = max_per_row.mean()
        
        # Classify
        if diag_sum > 0.3:
            pattern = 'sequential'
        elif max_focus > 0.5:
            pattern = 'variable_focused'
        elif entropy > 0.7:
            pattern = 'broad'
        else:
            pattern = 'mixed'
        
        return {
            'pattern_type': pattern,
            'diagonal_score': float(diag_sum),
            'entropy': float(entropy),
            'max_focus': float(max_focus),
            'interpretation': self._interpret_pattern(pattern)
        }
    
    def _interpret_pattern(self, pattern: str) -> str:
        """Provide human-readable interpretation."""
        interpretations = {
            'sequential': "Model attends to nearby code sequentially, suggesting local syntax/context focus",
            'variable_focused': "Model focuses on specific tokens, likely tracking variables or key identifiers",
            'broad': "Model distributes attention broadly, considering global context",
            'mixed': "Model combines multiple attention strategies"
        }
        return interpretations.get(pattern, "Unknown pattern")
    
    def get_attention_summary(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Get a summary of attention patterns for reporting.
        """
        attention_data = self.extract_attention(
            token_ids, line_positions, preceding_mask, attention_mask
        )
        
        summary = {}
        
        for name, attention in attention_data.items():
            pattern_info = self.classify_attention_pattern(attention)
            summary[name] = pattern_info
        
        return summary
