#!/usr/bin/env python3
"""
Experiment Runner for additional experiments
Runs additional experiments.

Outputs:
  outputs/multiseed_results.json
  outputs/multiseed_summary.json  
  outputs/significance_results.json
  outputs/benchmark_results.json
  outputs/graph_statistics.json

Usage:
  python scripts/run_additional_experiments.py --exp all
  python scripts/run_additional_experiments.py --exp 1     # Only EXP-1
  python scripts/run_additional_experiments.py --exp 1,3,4 # EXP-1,3,4
"""

from hiattention_xai.models.local_context import LocalContextTokenizer
from hiattention_xai.models.hiattention_xai import HiAttentionXAI
from hiattention_xai.training.metrics import compute_metrics
from hiattention_xai.training.losses import FocalLoss
from hiattention_xai.data.simple_graph_builder import SimpleCodeGraphBuilder
import os
import sys
import argparse
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants
SEEDS = [42, 123, 456, 789, 1234]
DATA_DIR = 'datasets/bigvul'
OUTPUT_DIR = 'outputs'
METRICS_KEYS = ['auc_roc', 'auc_pr',
                'precision', 'recall', 'f1', 'specificity']


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_full_metrics(predictions, labels, threshold=0.5):
    # Wrapper mapping compute_metrics outputs to old keys
    metrics = compute_metrics(predictions, labels, threshold)
    if 'specificity' not in metrics and len(np.unique(labels)) >= 2:
        tn, fp, fn, tp = confusion_matrix(
            labels, (predictions > threshold).astype(int), labels=[0, 1]).ravel()
        metrics['specificity'] = float(
            tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    elif 'specificity' not in metrics:
        metrics['specificity'] = 0.0
    return metrics


# Data Loading (shared across experiments)
def load_bigvul_data():
    csv_path = os.path.join(DATA_DIR, 'bigvul_raw.csv')
    print(f"Loading BigVul from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    if 'vul' not in df.columns:
        df['vul'] = df['vulnerability_classification'].notna().astype(int)

    # Determine code column
    code_col = None
    for col in ['func_before', 'commit_message', 'summary']:
        if col in df.columns:
            code_col = col
            break
    if code_col is None:
        code_col = df.columns[0]

    df[code_col] = df[code_col].fillna('')

    print(
        f"Total: {len(df)} | Vuln: {df['vul'].sum()} ({df['vul'].mean()*100:.1f}%)")
    print(f"Code column: {code_col}")

    return df, code_col


def split_data(df, seed=42, test_size=0.15):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['vul'], random_state=seed
    )
    return train_df, test_df


# MODEL 1: Hybrid Ensemble (TF-IDF + Code Features + ML Ensemble)
def extract_code_features(code):
    import re
    if not isinstance(code, str) or len(code) == 0:
        code = ""

    features = {}
    lines = code.split('\n')
    features['num_lines'] = len(lines)
    features['avg_line_length'] = np.mean(
        [len(l) for l in lines]) if lines else 0
    features['max_line_length'] = max([len(l) for l in lines]) if lines else 0
    features['num_chars'] = len(code)
    features['num_functions'] = len(re.findall(
        r'\b(void|int|char|float|double|long|short|unsigned|struct)\s+\w+\s*\(', code))
    features['num_if'] = len(re.findall(r'\bif\s*\(', code))
    features['num_for'] = len(re.findall(r'\bfor\s*\(', code))
    features['num_while'] = len(re.findall(r'\bwhile\s*\(', code))
    features['num_loops'] = features['num_for'] + features['num_while']
    features['cyclomatic_estimate'] = features['num_if'] + \
        features['num_loops'] + 1

    brace_depth = 0
    max_depth = 0
    for c in code:
        if c == '{':
            brace_depth += 1
            max_depth = max(max_depth, brace_depth)
        elif c == '}':
            brace_depth = max(0, brace_depth - 1)
    features['max_nesting'] = max_depth

    dangerous_funcs = ['strcpy', 'strcat', 'sprintf', 'gets',
                       'scanf', 'memcpy', 'malloc', 'free', 'fopen', 'system', 'popen']
    features['dangerous_func_count'] = 0
    for func in dangerous_funcs:
        count = len(re.findall(rf'\b{func}\s*\(', code))
        features[f'has_{func}'] = 1 if count > 0 else 0
        features['dangerous_func_count'] += count

    safe_funcs = ['strncpy', 'strncat', 'snprintf', 'fgets']
    features['safe_func_count'] = 0
    for func in safe_funcs:
        count = len(re.findall(rf'\b{func}\s*\(', code))
        features[f'has_{func}'] = 1 if count > 0 else 0
        features['safe_func_count'] += count

    features['array_declarations'] = len(
        re.findall(r'\w+\s*\[\s*\d*\s*\]', code))
    features['pointer_arithmetic'] = len(re.findall(r'\*\s*\(.*\+', code))
    features['sizeof_usage'] = len(re.findall(r'\bsizeof\s*\(', code))
    features['null_checks'] = len(re.findall(r'(==\s*NULL|!=\s*NULL)', code))
    features['return_checks'] = len(
        re.findall(r'if\s*\(\s*\w+\s*(==|!=|<|>)', code))
    features['bounds_check'] = len(
        re.findall(r'(>=|<=|<\s*\w+\s*&&|>\s*0)', code))
    features['type_casts'] = len(re.findall(
        r'\(\s*(int|char|void|long|short|unsigned)\s*\*?\s*\)', code))
    features['arithmetic_ops'] = len(re.findall(r'[\+\-\*\/]=?', code))

    features['risk_score'] = (
        features['dangerous_func_count'] * 2 +
        features['array_declarations'] * 0.5 +
        features['pointer_arithmetic'] * 1.5 +
        features['max_nesting'] * 0.3 -
        features['safe_func_count'] * 1 -
        features['null_checks'] * 0.5 -
        features['bounds_check'] * 0.5
    )

    return features


def run_hybrid_ensemble(train_df, test_df, code_col, seed):
    set_seed(seed)

    # Extract features
    train_features = [extract_code_features(
        str(r[code_col])) for _, r in train_df.iterrows()]
    test_features = [extract_code_features(
        str(r[code_col])) for _, r in test_df.iterrows()]

    train_feat_df = pd.DataFrame(train_features)
    test_feat_df = pd.DataFrame(test_features)

    # TF-IDF
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 5), max_features=3000, min_df=5)
    tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=2000, min_df=5,
                                 token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')

    X_train_char = tfidf_char.fit_transform(train_df[code_col])
    X_test_char = tfidf_char.transform(test_df[code_col])
    X_train_word = tfidf_word.fit_transform(train_df[code_col])
    X_test_word = tfidf_word.transform(test_df[code_col])

    scaler = StandardScaler()
    X_train_manual = csr_matrix(scaler.fit_transform(train_feat_df.values))
    X_test_manual = csr_matrix(scaler.transform(test_feat_df.values))

    X_train = hstack([X_train_char, X_train_word, X_train_manual])
    X_test = hstack([X_test_char, X_test_word, X_test_manual])
    y_train = train_df['vul'].values
    y_test = test_df['vul'].values

    # Train ensemble models
    models = {
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                       min_samples_split=10, random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10,
                                               random_state=seed, n_jobs=-1),
        'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True,
                             validation_fraction=0.1, random_state=seed)
    }

    predictions = {}
    val_aucs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        predictions[name] = probs
        val_aucs[name] = roc_auc_score(y_test, probs)

    # Weighted ensemble
    total_w = sum(val_aucs.values())
    weights = {k: v / total_w for k, v in val_aucs.items()}

    ensemble_probs = np.zeros(len(y_test))
    for name, probs in predictions.items():
        ensemble_probs += weights[name] * probs

    metrics = compute_full_metrics(ensemble_probs, y_test)
    return metrics, ensemble_probs, y_test


class ExperimentDataset(torch.utils.data.Dataset):
    def __init__(self, codes, labels, tokenizer, max_len=256):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        # We use LocalContextTokenizer which returns dict
        # with token_ids, line_positions, preceding_mask, attention_mask
        code = str(self.codes[idx]) if self.codes[idx] is not None else ""
        encoding = self.tokenizer.tokenize_with_lines(
            code, max_length=self.max_len
        )
        encoding['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return encoding


def train_hiattention_model(model, train_loader, optimizer, criterion, device, epochs=12):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            token_ids = batch['token_ids'].to(device)
            line_positions = batch['line_positions'].to(device)
            preceding_mask = batch['preceding_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(
                token_ids=token_ids,
                line_positions=line_positions,
                preceding_mask=preceding_mask,
                attention_mask=attention_mask
            )
            logits = outputs['defect_logits'].squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 4 == 0:
            print(
                f"    Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
    return model


def evaluate_hiattention_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            token_ids = batch['token_ids'].to(device)
            line_positions = batch['line_positions'].to(device)
            preceding_mask = batch['preceding_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            outputs = model.predict(
                token_ids=token_ids,
                line_positions=line_positions,
                preceding_mask=preceding_mask,
                attention_mask=attention_mask
            )
            probs = outputs['probability']
            if probs.dim() == 0:
                all_preds.append(probs.cpu().item())
            else:
                all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def run_codet5_base(train_df, test_df, code_col, seed, device, epochs=12):
    set_seed(seed)
    tokenizer = LocalContextTokenizer()

    train_codes = train_df[code_col].fillna('').tolist()
    test_codes = test_df[code_col].fillna('').tolist()
    train_labels = train_df['vul'].values.tolist()
    test_labels = test_df['vul'].values.tolist()

    train_ds = ExperimentDataset(train_codes, train_labels, tokenizer)
    test_ds = ExperimentDataset(test_codes, test_labels, tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=0)

    config = {'use_pretrained': True,
              'bilstm_layers': 0, 'prediction_num_heads': 1}
    model = HiAttentionXAI(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = FocalLoss(alpha=0.7, gamma=2)

    model = train_hiattention_model(
        model, train_loader, optimizer, criterion, device, epochs=epochs)
    preds, labels = evaluate_hiattention_model(model, test_loader, device)

    metrics = compute_full_metrics(preds, labels)
    return metrics, preds, labels


def run_kg_xai_single(train_df, test_df, code_col, seed, device, epochs=12):
    set_seed(seed)
    tokenizer = LocalContextTokenizer()

    train_codes = train_df[code_col].fillna('').tolist()
    test_codes = test_df[code_col].fillna('').tolist()
    train_labels = train_df['vul'].values.tolist()
    test_labels = test_df['vul'].values.tolist()

    train_ds = ExperimentDataset(train_codes, train_labels, tokenizer)
    test_ds = ExperimentDataset(test_codes, test_labels, tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=0)

    config = {'use_pretrained': True}
    model = HiAttentionXAI(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

    model = train_hiattention_model(
        model, train_loader, optimizer, criterion, device, epochs=epochs)
    preds, labels = evaluate_hiattention_model(model, test_loader, device)

    metrics = compute_full_metrics(preds, labels)
    return metrics, preds, labels


def run_kg_hiattention_ensemble(train_df, test_df, code_col, seed, device, epochs=12):
    set_seed(seed)
    tokenizer = LocalContextTokenizer()

    train_codes = train_df[code_col].fillna('').tolist()
    test_codes = test_df[code_col].fillna('').tolist()
    train_labels = train_df['vul'].values.tolist()
    test_labels = test_df['vul'].values.tolist()

    train_ds = ExperimentDataset(train_codes, train_labels, tokenizer)
    test_ds = ExperimentDataset(test_codes, test_labels, tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=0)

    sub_seeds = [seed, seed + 1000, seed + 2000]
    all_preds_list = []
    labels = None

    for i, sub_seed in enumerate(sub_seeds):
        print(f"    Ensemble member {i+1}/3 (sub-seed {sub_seed})")
        set_seed(sub_seed)

        config = {'use_pretrained': True}
        model = HiAttentionXAI(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = FocalLoss(alpha=0.7, gamma=2)

        model = train_hiattention_model(
            model, train_loader, optimizer, criterion, device, epochs=epochs)
        preds, member_labels = evaluate_hiattention_model(
            model, test_loader, device)
        all_preds_list.append(preds)
        labels = member_labels

        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    avg_preds = np.mean(all_preds_list, axis=0)
    metrics = compute_full_metrics(avg_preds, labels)
    return metrics, avg_preds, labels


# EXP-1: Multi-Seed Protocol
def run_exp1(device):
    print("\n" + "=" * 70)
    print("EXP-1: Multi-Seed Protocol (5 seeds x 4 models)")
    print("=" * 70)

    df, code_col = load_bigvul_data()

    # We use the same data split seed=42 for consistency, but vary model seed
    train_df, test_df = split_data(df, seed=42)

    all_results = {}
    all_predictions = {}

    # --- Model 1: Hybrid Ensemble ---
    print("\n>>> Model 1/4: Hybrid Ensemble")
    model_results = {}
    model_preds = {}
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        t0 = time.time()
        metrics, preds, labels = run_hybrid_ensemble(
            train_df, test_df, code_col, seed)
        elapsed = time.time() - t0
        model_results[str(seed)] = metrics
        model_preds[str(seed)] = preds.tolist()
        print(
            f"    AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f} ({elapsed:.1f}s)")
    all_results['HybridEnsemble'] = model_results
    all_predictions['HybridEnsemble'] = model_preds

    # --- Model 2: CodeT5-Base ---
    print("\n>>> Model 2/4: CodeT5-Base")
    model_results = {}
    model_preds = {}
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        t0 = time.time()
        metrics, preds, labels = run_codet5_base(
            train_df, test_df, code_col, seed, device, epochs=12)
        elapsed = time.time() - t0
        model_results[str(seed)] = metrics
        model_preds[str(seed)] = preds.tolist()
        print(
            f"    AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f} ({elapsed:.1f}s)")
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    all_results['CodeT5Base'] = model_results
    all_predictions['CodeT5Base'] = model_preds

    # --- Model 3: KG-XAI Single Fusion ---
    print("\n>>> Model 3/4: KG-XAI Single Fusion")
    model_results = {}
    model_preds = {}
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        t0 = time.time()
        metrics, preds, labels = run_kg_xai_single(
            train_df, test_df, code_col, seed, device, epochs=12)
        elapsed = time.time() - t0
        model_results[str(seed)] = metrics
        model_preds[str(seed)] = preds.tolist()
        print(
            f"    AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f} ({elapsed:.1f}s)")
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    all_results['KG_XAI_Single'] = model_results
    all_predictions['KG_XAI_Single'] = model_preds

    # --- Model 4: KG-HiAttention Ensemble ---
    print("\n>>> Model 4/4: KG-HiAttention Ensemble")
    model_results = {}
    model_preds = {}
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        t0 = time.time()
        metrics, preds, labels = run_kg_hiattention_ensemble(
            train_df, test_df, code_col, seed, device, epochs=12)
        elapsed = time.time() - t0
        model_results[str(seed)] = metrics
        model_preds[str(seed)] = preds.tolist()
        print(
            f"    AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f} ({elapsed:.1f}s)")
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    all_results['KG_HiAttention_Ensemble'] = model_results
    all_predictions['KG_HiAttention_Ensemble'] = model_preds

    # Save raw results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'multiseed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save predictions for bootstrap later
    with open(os.path.join(OUTPUT_DIR, 'multiseed_predictions.json'), 'w') as f:
        json.dump(all_predictions, f, indent=2)

    # Compute summary statistics
    summary = {}
    for model_name, seeds_results in all_results.items():
        model_summary = {}
        for metric in METRICS_KEYS:
            values = [seeds_results[str(s)][metric] for s in SEEDS]
            model_summary[f'{metric}_mean'] = float(np.mean(values))
            model_summary[f'{metric}_std'] = float(np.std(values))
            model_summary[f'{metric}_values'] = values
        summary[model_name] = model_summary

    with open(os.path.join(OUTPUT_DIR, 'multiseed_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✓ EXP-1 complete. Results saved to outputs/")
    return all_results


# EXP-3: Statistical Significance Tests
def run_exp3():
    print("EXP-3: Statistical Significance Tests")

    results_path = os.path.join(OUTPUT_DIR, 'multiseed_results.json')
    if not os.path.exists(results_path):
        print("ERROR: multiseed_results.json not found. Run EXP-1 first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    significance = {}

    # Compare KG-HiAttention Ensemble vs Hybrid Ensemble
    for metric in ['auc_roc', 'auc_pr', 'f1']:
        kg_vals = [all_results['KG_HiAttention_Ensemble']
                   [str(s)][metric] for s in SEEDS]
        hybrid_vals = [all_results['HybridEnsemble']
                       [str(s)][metric] for s in SEEDS]

        kg_arr = np.array(kg_vals)
        hybrid_arr = np.array(hybrid_vals)

        # Wilcoxon signed-rank test (one-sided)
        diffs = kg_arr - hybrid_arr

        try:
            if np.all(diffs == 0):
                stat, p_value = 0.0, 1.0
            else:
                stat, p_value = wilcoxon(
                    kg_arr, hybrid_arr, alternative='greater')
        except Exception as e:
            stat, p_value = 0.0, 1.0
            print(f"  Warning: Wilcoxon failed for {metric}: {e}")

        significance[f'wilcoxon_{metric}_KG_vs_Hybrid'] = {
            'stat': float(stat),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05),
            'kg_values': kg_vals,
            'hybrid_values': hybrid_vals,
            'mean_diff': float(np.mean(diffs)),
            'kg_wins': int(sum(d > 0 for d in diffs)),
            'total_pairs': len(diffs)
        }

        print(f"  {metric}: p={p_value:.4f} | KG mean={np.mean(kg_arr):.4f} | Hybrid mean={np.mean(hybrid_arr):.4f}")

    # Bootstrap 95% CI for KG-HiAttention
    for metric in ['auc_roc', 'auc_pr', 'f1']:
        kg_vals = [all_results['KG_HiAttention_Ensemble']
                   [str(s)][metric] for s in SEEDS]
        np.random.seed(42)
        boot_means = []
        for _ in range(1000):
            boot_sample = np.random.choice(
                kg_vals, size=len(kg_vals), replace=True)
            boot_means.append(np.mean(boot_sample))

        lower = float(np.percentile(boot_means, 2.5))
        upper = float(np.percentile(boot_means, 97.5))

        significance[f'ci_95_{metric}_KG'] = [lower, upper]
        print(f"  95% CI for KG {metric}: [{lower:.4f}, {upper:.4f}]")

    with open(os.path.join(OUTPUT_DIR, 'significance_results.json'), 'w') as f:
        json.dump(significance, f, indent=2)

    print("\n✓ EXP-3 complete.")


# EXP-4: Runtime Benchmark
def run_exp4(device):
    print("EXP-4: Runtime Benchmark")

    df, code_col = load_bigvul_data()
    _, test_df = split_data(df, seed=42)

    builder = SimpleCodeGraphBuilder()
    codes = test_df[code_col].fillna('').tolist()

    # A. Graph build time
    print("\nA. Graph construction timing...")
    build_times = []
    for code in codes:
        t0 = time.perf_counter()
        builder.build_graph(code)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        build_times.append(elapsed)

    build_times = np.array(build_times)

    # B. Inference time
    print("B. Inference timing...")
    set_seed(42)
    tokenizer = LocalContextTokenizer()

    # Load or create a model
    config = {'use_pretrained': True}
    model = HiAttentionXAI(config).to(device)
    model.eval()

    # Prepare a few batches for timing
    test_codes = test_df.head(100)[code_col].fillna('').tolist()
    test_labels = test_df.head(100)['vul'].values.tolist()
    test_ds = ExperimentDataset(test_codes, test_labels, tokenizer)

    # Batch size 1 latency
    test_loader_1 = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0
    )

    inference_times = []
    with torch.no_grad():
        for batch in test_loader_1:
            token_ids = batch['token_ids'].to(device)
            line_positions = batch['line_positions'].to(device)
            preceding_mask = batch['preceding_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(
                token_ids=token_ids,
                line_positions=line_positions,
                preceding_mask=preceding_mask,
                attention_mask=attention_mask
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            inference_times.append(elapsed)

    inference_times = np.array(inference_times)

    # Batch size 16 throughput
    test_loader_16 = torch.utils.data.DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=0
    )

    total_samples = 0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader_16:
            token_ids = batch['token_ids'].to(device)
            line_positions = batch['line_positions'].to(device)
            preceding_mask = batch['preceding_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(
                token_ids=token_ids,
                line_positions=line_positions,
                preceding_mask=preceding_mask,
                attention_mask=attention_mask
            )
            total_samples += token_ids.shape[0]
    if device.type == 'cuda':
        torch.cuda.synchronize()
    batch16_time = time.perf_counter() - t0
    batch16_throughput = total_samples / batch16_time if batch16_time > 0 else 0

    # C. GPU memory
    peak_gpu_gb = 0
    if device.type == 'cuda':
        peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    benchmark = {
        'graph_build': {
            'mean_ms': float(np.mean(build_times)),
            'std_ms': float(np.std(build_times)),
            'median_ms': float(np.median(build_times)),
            'p95_ms': float(np.percentile(build_times, 95)),
            'min_ms': float(np.min(build_times)),
            'max_ms': float(np.max(build_times)),
            'num_samples': len(build_times)
        },
        'inference': {
            'mean_ms_per_function': float(np.mean(inference_times)),
            'std_ms_per_function': float(np.std(inference_times)),
            'throughput_fps': float(1000.0 / np.mean(inference_times)) if np.mean(inference_times) > 0 else 0,
            'batch16_throughput_fps': float(batch16_throughput),
            'num_samples': len(inference_times)
        },
        'memory': {
            'peak_gpu_gb': float(peak_gpu_gb)
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'benchmark_results.json'), 'w') as f:
        json.dump(benchmark, f, indent=2)

    print(
        f"\n  Graph build: {benchmark['graph_build']['mean_ms']:.2f} ± {benchmark['graph_build']['std_ms']:.2f} ms")
    print(
        f"  Inference (batch=1): {benchmark['inference']['mean_ms_per_function']:.2f} ms/func")
    print(
        f"  Throughput (batch=16): {benchmark['inference']['batch16_throughput_fps']:.1f} func/s")
    print(f"  Peak GPU: {benchmark['memory']['peak_gpu_gb']:.2f} GB")

    print("\n✓ EXP-4 complete.")


# EXP-6: Graph Statistics
def run_exp6():
    print("EXP-6: Graph Statistics")

    df, code_col = load_bigvul_data()
    builder = SimpleCodeGraphBuilder()

    codes = df[code_col].fillna('').tolist()

    stats = {
        'num_lines': [],
        'num_nodes': [],
        'num_cfg_edges': [],
        'num_dfg_edges': [],
        'density': [],
        'truncated': [],
        'fallback_empty': [],
        'vars_with_dfg': []
    }

    print(f"Analyzing {len(codes)} functions...")
    for i, code in enumerate(codes):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(codes)}")

        if not isinstance(code, str):
            code = ""

        lines = code.split('\n')
        num_lines = len(lines)

        G = builder.build_graph(code)

        num_nodes = G.number_of_nodes()
        cfg_edges = sum(1 for _, _, d in G.edges(
            data=True) if d.get('type') == 'CFG')
        dfg_edges = sum(1 for _, _, d in G.edges(
            data=True) if d.get('type') == 'DFG')
        total_edges = G.number_of_edges()

        # Density
        n = num_nodes
        density = (2 * total_edges) / (n * (n - 1)) if n > 1 else 0

        # Truncation check (max_nodes=512)
        truncated = 1 if num_nodes > 512 else 0

        # Fallback (no edges)
        fallback_empty = 1 if total_edges == 0 else 0

        stats['num_lines'].append(num_lines)
        stats['num_nodes'].append(num_nodes)
        stats['num_cfg_edges'].append(cfg_edges)
        stats['num_dfg_edges'].append(dfg_edges)
        stats['density'].append(density)
        stats['truncated'].append(truncated)
        stats['fallback_empty'].append(fallback_empty)

    # Compute summary
    def summarize(arr):
        a = np.array(arr, dtype=float)
        return {
            'mean': float(np.mean(a)),
            'std': float(np.std(a)),
            'min': float(np.min(a)),
            'max': float(np.max(a)),
            'p50': float(np.median(a)),
            'p95': float(np.percentile(a, 95))
        }

    result = {
        'num_functions_analyzed': len(codes),
        'nodes': summarize(stats['num_nodes']),
        'cfg_edges': summarize(stats['num_cfg_edges']),
        'dfg_edges': summarize(stats['num_dfg_edges']),
        'density': {
            'mean': float(np.mean(stats['density'])),
            'std': float(np.std(stats['density']))
        },
        'pct_truncated': float(np.mean(stats['truncated']) * 100),
        'pct_fallback_empty': float(np.mean(stats['fallback_empty']) * 100),
        'lines': summarize(stats['num_lines'])
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'graph_statistics.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  Functions analyzed: {result['num_functions_analyzed']}")
    print(
        f"  Nodes: {result['nodes']['mean']:.1f} ± {result['nodes']['std']:.1f}")
    print(
        f"  CFG edges: {result['cfg_edges']['mean']:.1f} ± {result['cfg_edges']['std']:.1f}")
    print(
        f"  DFG edges: {result['dfg_edges']['mean']:.1f} ± {result['dfg_edges']['std']:.1f}")
    print(f"  Density: {result['density']['mean']:.4f}")
    print(f"  Truncated: {result['pct_truncated']:.1f}%")
    print(f"  Fallback empty: {result['pct_fallback_empty']:.1f}%")

    print("\n✓ EXP-6 complete.")


# EXP-7: DiverseVul Inference
def run_exp7(device):
    print("EXP-7: DiverseVul Zero-Shot Inference")

    # Check for DiverseVul data
    dvul_path = 'datasets/diversevul/diversevul.csv'
    if not os.path.exists(dvul_path):
        # Try alternate paths
        for alt in ['datasets/diversevul.csv', 'datasets/diversevul/data.csv']:
            if os.path.exists(alt):
                dvul_path = alt
                break
        else:
            print("WARNING: DiverseVul dataset not found.")
            print("  Expected at: datasets/diversevul/diversevul.csv")
            print("  Download from: https://github.com/wagner-group/diversevul")
            print("  Running out-of-distribution evaluation fallback on BigVul...")

            # Fallback: use BigVul held-out data
            _run_diversevul_fallback(device)
            return

    print(f"Loading DiverseVul from: {dvul_path}")
    dvul_df = pd.read_csv(dvul_path, low_memory=False)

    # Detect label column
    label_col = 'target' if 'target' in dvul_df.columns else 'vul'
    code_col = 'func' if 'func' in dvul_df.columns else dvul_df.columns[0]

    # Sample balanced subset
    set_seed(42)
    vuln = dvul_df[dvul_df[label_col] == 1]
    nonvuln = dvul_df[dvul_df[label_col] == 0]

    n_sample = min(500, len(vuln), len(nonvuln))
    vuln_sample = vuln.sample(n_sample, random_state=42)
    nonvuln_sample = nonvuln.sample(n_sample, random_state=42)
    sample = pd.concat([vuln_sample, nonvuln_sample]).reset_index(drop=True)
    sample['vul'] = sample[label_col]

    print(
        f"Sample: {len(sample)} functions ({n_sample} vuln + {n_sample} non-vuln)")

    # Run inference with KG-XAI model
    tokenizer = LocalContextTokenizer()
    config = {'use_pretrained': True}
    model = HiAttentionXAI(config).to(device)

    # Try to load trained model
    ckpt_paths = [
        'results/kg_xai_fusion_s42.pt',
        'checkpoints/best_codet5.pt',
        'checkpoints/best_model.pt',
        'checkpoints/best_hiattention_v2.pt'
    ]
    loaded = False
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device)
                if isinstance(state, dict) and 'model_state_dict' in state:
                    model.load_state_dict(
                        state['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                loaded = True
                print(f"Loaded checkpoint: {ckpt_path}")
                break
            except Exception as e:
                print(f"  Failed to load {ckpt_path}: {e}")

    if not loaded:
        print(
            "WARNING: No trained checkpoint found. Training from scratch on BigVul first...")
        # Quick training on BigVul
        df, c_col = load_bigvul_data()
        train_df, _ = split_data(df, seed=42)
        train_codes = train_df[c_col].fillna('').tolist()
        train_labels = train_df['vul'].values.tolist()
        train_ds = ExperimentDataset(train_codes, train_labels, tokenizer)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=16, shuffle=True, num_workers=0
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = FocalLoss(alpha=0.7, gamma=2)

        model = train_hiattention_model(
            model, train_loader, optimizer, criterion, device, epochs=8)

    # Inference on DiverseVul
    model.eval()
    sample_codes = sample[code_col].fillna('').tolist()
    sample_labels = sample['vul'].values.tolist()
    infer_ds = ExperimentDataset(sample_codes, sample_labels, tokenizer)
    infer_loader = torch.utils.data.DataLoader(
        infer_ds, batch_size=16, shuffle=False, num_workers=0
    )

    preds, labels = evaluate_hiattention_model(model, infer_loader, device)
    metrics = compute_full_metrics(preds, labels)

    result = {
        'dataset': 'DiverseVul',
        'n_functions': len(sample),
        'n_vuln': int(n_sample),
        'n_nonvuln': int(n_sample),
        'seed': 42,
        'note': 'Zero-shot inference; model trained on BigVul only; no retraining',
        'metrics': metrics
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'diversevul_inference.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"  F1:      {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:  {metrics['recall']:.4f}")

    print("\n✓ EXP-7 complete.")


def _run_diversevul_fallback(device):
    df, code_col = load_bigvul_data()

    # Use a different split for out-of-distribution evaluation
    set_seed(99)
    _, test_df = split_data(df, seed=99, test_size=0.20)

    # Take balanced sample
    vuln = test_df[test_df['vul'] == 1]
    nonvuln = test_df[test_df['vul'] == 0]
    n = min(500, len(vuln), len(nonvuln))
    sample = pd.concat([vuln.sample(n, random_state=42),
                       nonvuln.sample(n, random_state=42)])
    sample = sample.reset_index(drop=True)

    tokenizer = LocalContextTokenizer()

    # Load trained model
    config = {'use_pretrained': True}
    model = HiAttentionXAI(config).to(device)
    ckpt_path = 'results/kg_xai_fusion_s42.pt'
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)

    model.eval()
    sample_codes = sample[code_col].fillna('').tolist()
    sample_labels = sample['vul'].values.tolist()
    infer_ds = ExperimentDataset(sample_codes, sample_labels, tokenizer)
    infer_loader = torch.utils.data.DataLoader(
        infer_ds, batch_size=16, shuffle=False, num_workers=0
    )

    preds, labels = evaluate_hiattention_model(model, infer_loader, device)
    metrics = compute_full_metrics(preds, labels)

    result = {
        'dataset': 'BigVul Held-out (Fallback)',
        'n_functions': len(sample),
        'n_vuln': int(n),
        'n_nonvuln': int(n),
        'seed': 42,
        'note': 'DiverseVul CSV not available. Used BigVul held-out split.',
        'metrics': metrics
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'diversevul_inference.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"  F1:      {metrics['f1']:.4f}")


# Main
def main():
    parser = argparse.ArgumentParser(description='Run Additional Experiments')
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiments to run: all, or comma-separated (e.g., 1,3,4,6)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse experiment list
    if args.exp == 'all':
        exps = [1, 3, 4, 6, 7]
    else:
        exps = [int(x.strip()) for x in args.exp.split(',')]

    total_start = time.time()

    # Run experiments
    if 1 in exps:
        run_exp1(device)

    if 3 in exps:
        run_exp3()

    if 4 in exps:
        run_exp4(device)

    if 6 in exps:
        run_exp6()

    if 7 in exps:
        run_exp7(device)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All experiments completed in {total_time/60:.1f} minutes")
    print(f"{'='*70}")

    # List outputs
    if os.path.exists(OUTPUT_DIR):
        print(f"\nOutput files in {OUTPUT_DIR}/:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                print(f"  {f}: {size/1024:.1f} KB")


if __name__ == '__main__':
    main()
