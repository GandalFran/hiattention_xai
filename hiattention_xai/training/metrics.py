"""
Metrics Module

Computes evaluation metrics for defect prediction:
- Standard metrics (precision, recall, F1, AUC)
- Effort-aware metrics (Recall@Top20%LOC, Effort@Top20%Recall)
- Fairness metrics (SPD, EOD, AOD)
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from typing import Dict, Optional, List, Tuple


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        predictions: [N] - Predicted probabilities
        labels: [N] - Ground truth labels (0 or 1)
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    # Binary predictions
    binary_preds = (predictions > threshold).astype(int)
    
    # Handle edge cases
    if len(np.unique(labels)) < 2:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': (binary_preds == labels).mean(),
            'auc_roc': 0.5,
            'auc_pr': 0.0
        }
    
    metrics = {
        'precision': precision_score(labels, binary_preds, zero_division=0),
        'recall': recall_score(labels, binary_preds, zero_division=0),
        'f1': f1_score(labels, binary_preds, zero_division=0),
        'accuracy': (binary_preds == labels).mean(),
        'auc_roc': roc_auc_score(labels, predictions),
        'auc_pr': average_precision_score(labels, predictions)
    }
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
    metrics['true_positive'] = int(tp)
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def effort_aware_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    top_percentages: List[float] = [0.1, 0.2, 0.3, 0.5]
) -> Dict[str, float]:
    """
    Compute effort-aware metrics for defect prediction.
    
    These metrics are crucial for practical deployment:
    - How many defects found when reviewing top X% of lines?
    - How much code to review to find X% of defects?
    
    Args:
        predictions: [N] - Predicted probabilities
        labels: [N] - Ground truth labels
        top_percentages: Percentages to evaluate
    
    Returns:
        Dictionary of effort-aware metrics
    """
    n = len(predictions)
    total_defects = labels.sum()
    
    if total_defects == 0:
        return {f'recall_at_top{int(p*100)}': 0.0 for p in top_percentages}
    
    # Sort by prediction score (descending)
    sorted_indices = np.argsort(-predictions)
    sorted_labels = labels[sorted_indices]
    cumsum_defects = np.cumsum(sorted_labels)
    
    metrics = {}
    
    # Recall@Top X% LOC (Effort → Defects found)
    # "If we review top X% of code by prediction, what % of defects do we find?"
    for p in top_percentages:
        top_k = max(1, int(n * p))
        defects_found = cumsum_defects[top_k - 1]
        recall_at_top = defects_found / total_defects
        metrics[f'recall_at_top{int(p*100)}'] = recall_at_top
    
    # Effort@Top X% Recall (Defects → Effort needed)
    # "How much code to review to find X% of defects?"
    for p in top_percentages:
        target_defects = total_defects * p
        effort_idx = np.argmax(cumsum_defects >= target_defects)
        effort = (effort_idx + 1) / n
        metrics[f'effort_at_{int(p*100)}recall'] = effort
    
    # IFA (Initial False Alarm) - number of clean lines before first defect
    first_defect_idx = np.argmax(sorted_labels > 0)
    metrics['ifa'] = first_defect_idx
    
    # PofB20 (Percentage of Bugs in top 20%)
    # Normalized version of recall@20
    top_20_idx = max(1, int(n * 0.2))
    metrics['pofb20'] = cumsum_defects[top_20_idx - 1] / total_defects
    
    return metrics


def compute_fairness_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    protected_attr: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute fairness metrics across protected attribute groups.
    
    Protected attributes could be: programming language, project, complexity level.
    
    Metrics:
    - SPD (Statistical Parity Difference): Difference in positive prediction rates
    - EOD (Equal Opportunity Difference): Difference in true positive rates
    - AOD (Average Odds Difference): Average of TPR and FPR differences
    
    Args:
        predictions: [N] - Predicted probabilities
        labels: [N] - Ground truth labels
        protected_attr: [N] - Group assignments
        threshold: Classification threshold
    
    Returns:
        Dictionary of fairness metrics
    """
    binary_preds = (predictions > threshold).astype(int)
    
    groups = np.unique(protected_attr)
    
    if len(groups) < 2:
        return {'spd': 0.0, 'eod': 0.0, 'aod': 0.0}
    
    group_metrics = {}
    
    for group in groups:
        mask = protected_attr == group
        group_preds = binary_preds[mask]
        group_labels = labels[mask]
        
        # Positive prediction rate
        ppr = group_preds.mean() if len(group_preds) > 0 else 0
        
        # True positive rate
        positives = group_labels == 1
        tpr = group_preds[positives].mean() if positives.sum() > 0 else 0
        
        # False positive rate
        negatives = group_labels == 0
        fpr = group_preds[negatives].mean() if negatives.sum() > 0 else 0
        
        group_metrics[group] = {
            'ppr': ppr,
            'tpr': tpr,
            'fpr': fpr,
            'count': mask.sum()
        }
    
    # Compute pairwise differences
    pprs = [g['ppr'] for g in group_metrics.values()]
    tprs = [g['tpr'] for g in group_metrics.values()]
    fprs = [g['fpr'] for g in group_metrics.values()]
    
    # SPD: Max difference in positive prediction rates
    spd = max(pprs) - min(pprs)
    
    # EOD: Max difference in TPR
    eod = max(tprs) - min(tprs)
    
    # AOD: Average of TPR and FPR differences
    fpr_diff = max(fprs) - min(fprs)
    aod = 0.5 * (eod + fpr_diff)
    
    return {
        'spd': spd,
        'eod': eod,
        'aod': aod,
        'fairness_status': 'FAIR' if spd < 0.1 and eod < 0.1 else 'BIASED',
        'group_metrics': group_metrics
    }


def optimal_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    method: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        predictions: [N] - Predicted probabilities
        labels: [N] - Ground truth labels
        method: Optimization target ('f1', 'youden', 'cost')
    
    Returns:
        (optimal_threshold, metric_value)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_value = 0.0
    
    for thresh in thresholds:
        binary = (predictions > thresh).astype(int)
        
        if method == 'f1':
            value = f1_score(labels, binary, zero_division=0)
        elif method == 'youden':
            # Youden's J statistic: sensitivity + specificity - 1
            tn, fp, fn, tp = confusion_matrix(labels, binary, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            value = sensitivity + specificity - 1
        elif method == 'cost':
            # Minimize cost (FN more expensive than FP)
            tn, fp, fn, tp = confusion_matrix(labels, binary, labels=[0, 1]).ravel()
            cost = fp + 5 * fn  # FN is 5x more costly
            value = -cost  # Negate to maximize
        else:
            value = f1_score(labels, binary, zero_division=0)
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return best_threshold, best_value


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval via bootstrap.
    
    Args:
        predictions: [N] - Predicted probabilities
        labels: [N] - Ground truth labels
        metric_fn: Function that takes (preds, labels) and returns scalar
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    n = len(predictions)
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_preds = predictions[indices]
        boot_labels = labels[indices]
        
        try:
            value = metric_fn(boot_preds, boot_labels)
            bootstrap_values.append(value)
        except Exception:
            continue
    
    bootstrap_values = np.array(bootstrap_values)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    mean = bootstrap_values.mean()
    
    return mean, lower, upper
