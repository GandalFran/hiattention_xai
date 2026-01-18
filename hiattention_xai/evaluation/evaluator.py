"""
Evaluation Framework

Comprehensive evaluation with SOTA comparisons and fairness analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from ..training.metrics import (
    compute_metrics, 
    effort_aware_metrics, 
    compute_fairness_metrics,
    optimal_threshold,
    bootstrap_confidence_interval
)


class EvaluationFramework:
    """
    Comprehensive evaluation framework for HiAttention-XAI.
    
    Features:
    - Standard and effort-aware metrics
    - Confidence intervals via bootstrap
    - Fairness analysis
    - SOTA comparison
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def evaluate(
        self,
        dataloader,
        threshold: float = 0.5,
        compute_ci: bool = False,
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        """
        self.model.eval()
        
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    token_ids=batch['token_ids'].to(self.device),
                    line_positions=batch['line_positions'].to(self.device),
                    preceding_mask=batch['preceding_mask'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                
                probs = outputs['defect_probability'].squeeze(-1).cpu().numpy()
                predictions.extend(probs)
                labels.extend(batch['label'].numpy())
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Standard metrics
        results = compute_metrics(predictions, labels, threshold)
        
        # Effort-aware metrics
        results.update(effort_aware_metrics(predictions, labels))
        
        # Optimal threshold
        opt_thresh, opt_f1 = optimal_threshold(predictions, labels, 'f1')
        results['optimal_threshold'] = opt_thresh
        results['optimal_f1'] = opt_f1
        
        # Confidence intervals
        if compute_ci:
            def f1_fn(p, l): 
                from sklearn.metrics import f1_score
                return f1_score(l, (p > threshold).astype(int), zero_division=0)
            
            mean, lower, upper = bootstrap_confidence_interval(
                predictions, labels, f1_fn, n_bootstrap
            )
            results['f1_ci'] = {'mean': mean, 'lower': lower, 'upper': upper}
        
        return results
    
    def evaluate_by_group(
        self,
        dataloader,
        group_fn,
        threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """Evaluate separately for different groups."""
        self.model.eval()
        
        group_predictions = {}
        group_labels = {}
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    token_ids=batch['token_ids'].to(self.device),
                    line_positions=batch['line_positions'].to(self.device),
                    preceding_mask=batch['preceding_mask'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                
                probs = outputs['defect_probability'].squeeze(-1).cpu().numpy()
                labs = batch['label'].numpy()
                
                for i, (p, l) in enumerate(zip(probs, labs)):
                    group = group_fn(batch, i)
                    if group not in group_predictions:
                        group_predictions[group] = []
                        group_labels[group] = []
                    group_predictions[group].append(p)
                    group_labels[group].append(l)
        
        results = {}
        for group in group_predictions:
            preds = np.array(group_predictions[group])
            labs = np.array(group_labels[group])
            results[group] = compute_metrics(preds, labs, threshold)
        
        return results


class FairnessAnalyzer:
    """Analyzes model fairness across protected attributes."""
    
    @staticmethod
    def analyze(
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attr: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """Run complete fairness analysis."""
        return compute_fairness_metrics(predictions, labels, protected_attr, threshold)
    
    @staticmethod
    def generate_report(fairness_results: Dict) -> str:
        """Generate fairness report."""
        lines = ["# Fairness Analysis Report\n"]
        
        lines.append(f"**Status**: {fairness_results['fairness_status']}\n")
        lines.append(f"- SPD: {fairness_results['spd']:.4f} (threshold: 0.1)")
        lines.append(f"- EOD: {fairness_results['eod']:.4f} (threshold: 0.1)")
        lines.append(f"- AOD: {fairness_results['aod']:.4f} (threshold: 0.1)\n")
        
        if 'group_metrics' in fairness_results:
            lines.append("## Group Metrics\n")
            for group, metrics in fairness_results['group_metrics'].items():
                lines.append(f"### {group}")
                lines.append(f"- Count: {metrics['count']}")
                lines.append(f"- PPR: {metrics['ppr']:.4f}")
                lines.append(f"- TPR: {metrics['tpr']:.4f}")
                lines.append(f"- FPR: {metrics['fpr']:.4f}\n")
        
        return '\n'.join(lines)


class SOTAComparison:
    """Compare results against state-of-the-art baselines."""
    
    # Known SOTA results from literature
    BASELINES = {
        'PLEASE': {
            'paper': 'Habib et al. 2024',
            'recall_at_top20': 0.67,
            'f1': 0.45,
            'auc_roc': 0.82
        },
        'LineVul': {
            'paper': 'Fu et al. 2022',
            'recall_at_top20': 0.58,
            'f1': 0.41,
            'auc_roc': 0.78
        },
        'DeepLineDP': {
            'paper': 'Pornprasit & Tantithamthavorn 2021',
            'recall_at_top20': 0.52,
            'f1': 0.38,
            'auc_roc': 0.74
        },
        'CodeBERT': {
            'paper': 'Feng et al. 2020',
            'recall_at_top20': 0.48,
            'f1': 0.35,
            'auc_roc': 0.71
        }
    }
    
    @classmethod
    def compare(cls, our_results: Dict) -> Dict:
        """Compare our results to baselines."""
        comparison = {'our_model': our_results, 'baselines': {}}
        
        for name, baseline in cls.BASELINES.items():
            comparison['baselines'][name] = {
                'paper': baseline['paper'],
                'metrics': {k: v for k, v in baseline.items() if k != 'paper'},
                'comparison': {}
            }
            
            for metric in ['recall_at_top20', 'f1', 'auc_roc']:
                if metric in our_results and metric in baseline:
                    diff = our_results[metric] - baseline[metric]
                    pct = (diff / baseline[metric]) * 100
                    comparison['baselines'][name]['comparison'][metric] = {
                        'ours': our_results[metric],
                        'baseline': baseline[metric],
                        'diff': diff,
                        'improvement_pct': pct
                    }
        
        return comparison
    
    @classmethod
    def generate_table(cls, comparison: Dict) -> str:
        """Generate comparison table."""
        lines = ["| Model | Recall@20% | F1 | AUC-ROC |", "|-------|------------|-----|---------|"]
        
        our = comparison['our_model']
        lines.append(f"| **HiAttention-XAI (Ours)** | **{our.get('recall_at_top20', 0):.3f}** | **{our.get('f1', 0):.3f}** | **{our.get('auc_roc', 0):.3f}** |")
        
        for name, data in comparison['baselines'].items():
            m = data['metrics']
            lines.append(f"| {name} | {m.get('recall_at_top20', 0):.3f} | {m.get('f1', 0):.3f} | {m.get('auc_roc', 0):.3f} |")
        
        return '\n'.join(lines)
