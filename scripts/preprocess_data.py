#!/usr/bin/env python3
"""
Data Preprocessing Script

Processes raw repositories into training-ready format.

Usage:
    python scripts/preprocess_data.py --input datasets/raw --output datasets/processed
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from hiattention_xai.data import DataPreprocessor, CodeParser, CodeGraphBuilder


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess code repositories')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Directory containing raw repositories')
    parser.add_argument('--output', type=str, required=True,
                       help='Directory for processed output')
    parser.add_argument('--max_repos', type=int, default=None,
                       help='Maximum number of repositories to process')
    parser.add_argument('--langs', type=str, nargs='+', default=['py', 'java', 'c'],
                       help='Languages to process')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("HiAttention-XAI Data Preprocessing")
    print("=" * 60)
    
    # Find repositories
    input_path = Path(args.input)
    repos = [d for d in input_path.iterdir() if d.is_dir()]
    
    if args.max_repos:
        repos = repos[:args.max_repos]
    
    print(f"\nFound {len(repos)} repositories to process")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        tokenizer_name='Salesforce/codet5-base',
        max_length=512,
        context_window=5
    )
    
    all_stats = []
    
    for i, repo_path in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] Processing: {repo_path.name}")
        
        output_dir = os.path.join(args.output, repo_path.name)
        
        try:
            stats = preprocessor.process_repository(
                str(repo_path),
                output_dir
            )
            stats['repo_name'] = repo_path.name
            all_stats.append(stats)
            
            print(f"  Functions: {stats.get('num_functions', 0)}")
            print(f"  Examples:  {stats.get('num_examples', 0)}")
            print(f"  Defects:   {stats.get('num_defective', 0)}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save combined statistics
    combined_stats = {
        'total_repos': len(all_stats),
        'total_examples': sum(s.get('num_examples', 0) for s in all_stats),
        'total_defective': sum(s.get('num_defective', 0) for s in all_stats),
        'total_functions': sum(s.get('num_functions', 0) for s in all_stats),
        'repos': all_stats
    }
    
    with open(os.path.join(args.output, 'combined_stats.json'), 'w') as f:
        json.dump(combined_stats, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Total repositories: {combined_stats['total_repos']}")
    print(f"Total examples:     {combined_stats['total_examples']}")
    print(f"Total defective:    {combined_stats['total_defective']}")
    print(f"Output directory:   {args.output}")


if __name__ == '__main__':
    main()
