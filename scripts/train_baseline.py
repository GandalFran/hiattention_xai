#!/usr/bin/env python3
"""
Baseline Model Training (TF-IDF + Logistic Regression)
For comparison with HiAttention-XAI in the paper.
"""

import os
import h5py
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

def load_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        # Convert token IDs back to space-separated strings for TF-IDF
        token_ids = f['token_ids'][:]
        texts = [" ".join(map(str, row[row > 0])) for row in token_ids]
        labels = f['labels'][:]
    return texts, labels

def main():
    print("="*60)
    print("Baseline Training: TF-IDF + LogReg")
    print("="*60)
    
    data_dir = "datasets/processed/strong_signal"
    
    print("Loading data...")
    X_train, y_train = load_data(os.path.join(data_dir, 'train.h5'))
    X_test, y_test = load_data(os.path.join(data_dir, 'test.h5'))
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")
    
    print("\nVectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression...")
    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_train_vec, y_train)
    
    print("Evaluating...")
    probs = clf.predict_proba(X_test_vec)[:, 1]
    preds = clf.predict(X_test_vec)
    
    metrics = {
        'auc_roc': float(roc_auc_score(y_test, probs)),
        'f1': float(f1_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds)),
        'recall': float(recall_score(y_test, preds))
    }
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("\nResults saved to results/baseline_results.json")

if __name__ == '__main__':
    main()
