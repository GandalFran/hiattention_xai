"""
Visualization Utilities

Plotting functions for training curves, metrics, and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_training_curves(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot training loss and validation metrics over epochs.
    
    Args:
        history: Training history dict with 'train_loss', 'val_metrics', etc.
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Training loss
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    if 'val_metrics' in history:
        val_losses = [m.get('val_loss', 0) for m in history['val_metrics']]
        ax1.plot(val_losses, 'r--', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 and Recall
    ax2 = axes[1]
    if 'val_metrics' in history:
        f1_scores = [m.get('f1', 0) for m in history['val_metrics']]
        recalls = [m.get('recall', 0) for m in history['val_metrics']]
        precisions = [m.get('precision', 0) for m in history['val_metrics']]
        
        ax2.plot(f1_scores, 'g-', linewidth=2, label='F1')
        ax2.plot(recalls, 'b-', linewidth=2, label='Recall')
        ax2.plot(precisions, 'r-', linewidth=2, label='Precision')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Classification Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Effort-aware metrics
    ax3 = axes[2]
    if 'val_metrics' in history:
        recall_top20 = [m.get('recall_at_top20', 0) for m in history['val_metrics']]
        ax3.plot(recall_top20, 'm-', linewidth=2, label='Recall@Top20%')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Effort-Aware Metrics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_heatmap(
    attention: np.ndarray,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Pattern",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot attention weights as heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Limit size
    max_size = 40
    if attention.shape[0] > max_size:
        attention = attention[:max_size, :max_size]
        if tokens:
            tokens = tokens[:max_size]
    
    sns.heatmap(
        attention,
        ax=ax,
        cmap='viridis',
        square=True,
        xticklabels=tokens if tokens else False,
        yticklabels=tokens if tokens else False,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_title(title, fontsize=14)
    
    if tokens:
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot confusion matrix.
    """
    from sklearn.metrics import confusion_matrix
    
    binary_preds = (predictions > threshold).astype(int)
    cm = confusion_matrix(labels, binary_preds, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        ax=ax,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Clean', 'Defective'],
        yticklabels=['Clean', 'Defective']
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot ROC curve.
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_fairness_comparison(
    fairness_metrics: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot fairness metrics comparison across groups.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = ['spd', 'eod', 'aod']
    titles = ['Statistical Parity Difference', 'Equal Opportunity Difference', 'Average Odds Difference']
    
    for ax, metric, title in zip(axes, metrics, titles):
        groups = list(fairness_metrics.keys())
        values = [fairness_metrics[g].get(metric, 0) for g in groups]
        
        bars = ax.bar(groups, values, color='steelblue', alpha=0.8)
        ax.axhline(y=0.1, color='r', linestyle='--', label='Fairness threshold')
        
        ax.set_xlabel('Protected Attribute')
        ax.set_ylabel('Difference')
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, max(0.3, max(values) * 1.1))
        
        # Color bars above threshold
        for bar, val in zip(bars, values):
            if val > 0.1:
                bar.set_color('salmon')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_token_importance(
    tokens: List[str],
    importance_scores: List[float],
    top_k: int = 20,
    title: str = "Token Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot token importance scores as horizontal bar chart.
    """
    # Get top-k tokens
    indices = np.argsort(-np.array(importance_scores))[:top_k]
    top_tokens = [tokens[i] for i in indices]
    top_scores = [importance_scores[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_tokens))
    colors = ['#d9534f' if s > 0 else '#5cb85c' for s in top_scores]
    
    ax.barh(y_pos, top_scores, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_tokens, fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
