"""
Report Generator - Comprehensive Explainability Module

Combines all explainability methods into human-readable reports.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any

from .saliency import SaliencyExplainer
from .shap_explainer import SHAPExplainer
from .attention_viz import AttentionVisualizer


class ExplainabilityModule:
    """
    Unified explainability interface combining all XAI methods.
    
    Generates comprehensive explanations for defect predictions including:
    - Saliency-based token importance
    - SHAP values
    - Attention pattern analysis
    - Human-readable report generation
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize sub-explainers
        self.saliency = SaliencyExplainer(model, tokenizer)
        self.shap = SHAPExplainer(model, tokenizer, device)
        self.attention = AttentionVisualizer(model, tokenizer)
    
    def explain(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        methods: List[str] = ['saliency', 'shap', 'attention'],
        n_shap_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            token_ids: [1, seq_len] - Input tokens
            line_positions: [1, seq_len] - Line positions
            preceding_mask: [1, seq_len] - Preceding line mask
            attention_mask: [1, seq_len] - Padding mask
            methods: Which explanation methods to use
            n_shap_samples: Number of samples for SHAP
        
        Returns:
            Comprehensive explanation dictionary
        """
        explanation = {
            'input_info': self._get_input_info(token_ids),
            'prediction': None,
            'explanations': {},
            'summary': None
        }
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                token_ids=token_ids.to(self.device),
                line_positions=line_positions.to(self.device),
                preceding_mask=preceding_mask.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
            )
        
        explanation['prediction'] = {
            'probability': outputs['defect_probability'].item(),
            'is_defective': outputs['defect_probability'].item() > 0.5,
            'epistemic_uncertainty': outputs.get('epistemic_uncertainty', torch.tensor(0)).item(),
            'aleatoric_uncertainty': outputs.get('aleatoric_uncertainty', torch.tensor(0)).item()
        }
        
        # Compute explanations from each method
        if 'saliency' in methods:
            saliency_scores, saliency_meta = self.saliency.compute_saliency(
                token_ids, line_positions, preceding_mask, attention_mask
            )
            explanation['explanations']['saliency'] = {
                'scores': saliency_scores.tolist(),
                'top_tokens': self.saliency.get_top_tokens(saliency_scores, token_ids),
                'metadata': saliency_meta
            }
        
        if 'shap' in methods:
            shap_result = self.shap.compute_shap_values(
                token_ids, line_positions, preceding_mask, attention_mask,
                n_samples=n_shap_samples
            )
            explanation['explanations']['shap'] = shap_result
        
        if 'attention' in methods:
            attention_summary = self.attention.get_attention_summary(
                token_ids, line_positions, preceding_mask, attention_mask
            )
            explanation['explanations']['attention'] = attention_summary
        
        # Generate summary
        explanation['summary'] = self._generate_summary(explanation)
        
        return explanation
    
    def _get_input_info(self, token_ids: torch.Tensor) -> Dict:
        """Extract input information."""
        seq_len = token_ids.size(1)
        
        info = {
            'num_tokens': seq_len,
            'unique_tokens': len(torch.unique(token_ids))
        }
        
        if self.tokenizer:
            info['code_preview'] = self.tokenizer.decode(
                token_ids[0][:100].tolist(),
                skip_special_tokens=True
            )[:200]  # First 200 chars
        
        return info
    
    def _generate_summary(self, explanation: Dict) -> Dict:
        """Generate human-readable summary."""
        pred = explanation['prediction']
        
        verdict = "DEFECTIVE" if pred['is_defective'] else "CLEAN"
        confidence = 1.0 / (1.0 + pred['epistemic_uncertainty'] + pred['aleatoric_uncertainty'])
        
        summary = {
            'verdict': verdict,
            'confidence': confidence,
            'probability': pred['probability']
        }
        
        # Find most important tokens across methods
        important_tokens = []
        
        if 'saliency' in explanation['explanations']:
            saliency_top = explanation['explanations']['saliency'].get('top_tokens', [])
            for t in saliency_top[:3]:
                important_tokens.append({
                    'method': 'saliency',
                    'token': t.get('token_string', f"token_{t['position']}"),
                    'score': t['score']
                })
        
        if 'shap' in explanation['explanations']:
            shap_pos = explanation['explanations']['shap'].get('top_positive', [])
            for t in shap_pos[:3]:
                important_tokens.append({
                    'method': 'shap',
                    'token': t.get('token', f"position_{t['position']}"),
                    'score': t['value']
                })
        
        summary['important_tokens'] = important_tokens
        
        # Attention pattern
        if 'attention' in explanation['explanations']:
            for attn_type, pattern in explanation['explanations']['attention'].items():
                summary['attention_pattern'] = pattern.get('pattern_type', 'unknown')
                break
        
        # Human-readable explanation
        summary['natural_language'] = self._generate_natural_language(summary, pred)
        
        return summary
    
    def _generate_natural_language(self, summary: Dict, pred: Dict) -> str:
        """Generate natural language explanation."""
        verdict = summary['verdict']
        prob = pred['probability'] * 100
        confidence = summary['confidence'] * 100
        
        if verdict == "DEFECTIVE":
            text = f"This code is predicted to be DEFECTIVE with {prob:.1f}% probability. "
        else:
            text = f"This code is predicted to be CLEAN with {100-prob:.1f}% confidence. "
        
        text += f"Model confidence: {confidence:.1f}%. "
        
        # Mention important tokens
        if summary['important_tokens']:
            tokens = [t['token'] for t in summary['important_tokens'][:3]]
            tokens_str = ', '.join([f"'{t}'" for t in tokens])
            text += f"Key tokens influencing this prediction: {tokens_str}. "
        
        # Mention attention pattern
        if 'attention_pattern' in summary:
            pattern = summary['attention_pattern']
            text += f"The model used a {pattern} attention pattern. "
        
        return text
    
    def generate_report(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_format: str = 'markdown'
    ) -> str:
        """
        Generate formatted report for the prediction.
        
        Args:
            output_format: 'markdown', 'html', or 'text'
        
        Returns:
            Formatted report string
        """
        explanation = self.explain(
            token_ids, line_positions, preceding_mask, attention_mask
        )
        
        if output_format == 'markdown':
            return self._format_markdown(explanation)
        elif output_format == 'html':
            return self._format_html(explanation)
        else:
            return self._format_text(explanation)
    
    def _format_markdown(self, explanation: Dict) -> str:
        """Format as Markdown."""
        lines = []
        
        pred = explanation['prediction']
        summary = explanation['summary']
        
        lines.append("# Defect Prediction Report\n")
        
        lines.append("## Prediction\n")
        lines.append(f"- **Verdict**: {summary['verdict']}")
        lines.append(f"- **Probability**: {pred['probability']*100:.1f}%")
        lines.append(f"- **Confidence**: {summary['confidence']*100:.1f}%")
        lines.append("")
        
        lines.append("## Explanation\n")
        lines.append(summary['natural_language'])
        lines.append("")
        
        lines.append("### Important Tokens\n")
        lines.append("| Token | Method | Score |")
        lines.append("|-------|--------|-------|")
        for t in summary.get('important_tokens', []):
            lines.append(f"| `{t['token']}` | {t['method']} | {t['score']:.3f} |")
        lines.append("")
        
        if 'attention_pattern' in summary:
            lines.append("### Attention Analysis\n")
            lines.append(f"Pattern type: **{summary['attention_pattern']}**")
        
        return '\n'.join(lines)
    
    def _format_text(self, explanation: Dict) -> str:
        """Format as plain text."""
        lines = []
        
        pred = explanation['prediction']
        summary = explanation['summary']
        
        lines.append("=" * 50)
        lines.append("DEFECT PREDICTION REPORT")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Verdict: {summary['verdict']}")
        lines.append(f"Probability: {pred['probability']*100:.1f}%")
        lines.append(f"Confidence: {summary['confidence']*100:.1f}%")
        lines.append("")
        lines.append("Explanation:")
        lines.append(summary['natural_language'])
        lines.append("")
        lines.append("Important Tokens:")
        for t in summary.get('important_tokens', []):
            lines.append(f"  - {t['token']} ({t['method']}): {t['score']:.3f}")
        
        return '\n'.join(lines)
    
    def _format_html(self, explanation: Dict) -> str:
        """Format as HTML."""
        pred = explanation['prediction']
        summary = explanation['summary']
        
        color = "#d9534f" if summary['verdict'] == "DEFECTIVE" else "#5cb85c"
        
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
            <h2>Defect Prediction Report</h2>
            <div style="background-color: {color}; color: white; padding: 10px; border-radius: 3px; margin-bottom: 15px;">
                <strong>{summary['verdict']}</strong> - {pred['probability']*100:.1f}% probability
            </div>
            <p>{summary['natural_language']}</p>
            <h3>Important Tokens</h3>
            <ul>
        """
        
        for t in summary.get('important_tokens', []):
            html += f"<li><code>{t['token']}</code> ({t['method']}): {t['score']:.3f}</li>"
        
        html += "</ul></div>"
        
        return html
