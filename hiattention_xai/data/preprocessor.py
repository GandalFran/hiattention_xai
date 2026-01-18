"""
Data Preprocessor Module

Complete pipeline from raw source code to training-ready format:
1. Parse code (AST)
2. Extract functions and build graphs
3. Label defects from git history or annotations
4. Create training examples
5. Tokenize and encode
"""

import os
import json
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import h5py

from .code_parser import CodeParser, FunctionInfo
from .graph_builder import CodeGraphBuilder


@dataclass
class TrainingExample:
    """A single training example for defect prediction."""
    example_id: str
    function_id: str
    line_number: int
    line_code: str
    token_ids: List[int]
    line_positions: List[int]
    preceding_mask: List[bool]
    attention_mask: List[int]
    label: int  # 0 = clean, 1 = defective
    context_lines: List[str]
    metadata: Dict[str, Any]


class DefectDataset(Dataset):
    """PyTorch Dataset for defect prediction training."""
    
    def __init__(
        self,
        examples: List[TrainingExample],
        max_length: int = 512
    ):
        self.examples = examples
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Pad or truncate to max_length
        def pad_or_truncate(seq, pad_value=0):
            if len(seq) >= self.max_length:
                return seq[:self.max_length]
            return seq + [pad_value] * (self.max_length - len(seq))
        
        token_ids = pad_or_truncate(example.token_ids, 0)
        line_positions = pad_or_truncate(example.line_positions, 0)
        preceding_mask = pad_or_truncate([int(m) for m in example.preceding_mask], 0)
        attention_mask = pad_or_truncate(example.attention_mask, 0)
        
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'line_positions': torch.tensor(line_positions, dtype=torch.long),
            'preceding_mask': torch.tensor(preceding_mask, dtype=torch.bool),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(example.label, dtype=torch.long),
            'example_id': example.example_id
        }


class DataPreprocessor:
    """
    Complete data preprocessing pipeline.
    
    Steps:
    1. Parse code repository
    2. Build function dependency graphs
    3. Extract defect labels from git history
    4. Create training examples with context
    5. Tokenize and encode
    """
    
    def __init__(
        self,
        tokenizer_name: str = "Salesforce/codet5-base",
        max_length: int = 512,
        context_window: int = 5
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.context_window = context_window
        
        self.parser: Optional[CodeParser] = None
        self.graph_builder: Optional[CodeGraphBuilder] = None
    
    def process_repository(
        self,
        repo_path: str,
        output_dir: str,
        defect_labels: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, Any]:
        """
        Process a repository and create training data.
        
        Args:
            repo_path: Path to repository root
            output_dir: Directory to save processed data
            defect_labels: Optional pre-existing labels {func_id: [defective_lines]}
        
        Returns:
            Statistics about processing
        """
        print(f"Processing repository: {repo_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Parse code
        print("  Step 1: Parsing code...")
        self.parser = CodeParser(repo_path)
        functions = self.parser.parse_repository()
        print(f"    Extracted {len(functions)} functions")
        
        # Step 2: Build graphs
        print("  Step 2: Building dependency graphs...")
        self.graph_builder = CodeGraphBuilder()
        self.graph_builder.build_from_parser(self.parser)
        graph_stats = self.graph_builder.get_statistics()
        print(f"    Built graph: {graph_stats['num_call_edges']} call edges")
        
        # Step 3: Load or extract defect labels
        print("  Step 3: Processing defect labels...")
        if defect_labels is None:
            defect_labels = self._extract_defects_from_git(repo_path)
        print(f"    Found defects in {len(defect_labels)} functions")
        
        # Step 4: Create training examples
        print("  Step 4: Creating training examples...")
        examples = self._create_examples(functions, defect_labels)
        print(f"    Created {len(examples)} examples")
        
        # Step 5: Save processed data
        print("  Step 5: Saving processed data...")
        self._save_dataset(examples, output_dir)
        
        # Save graph data
        graph_data = self.graph_builder.to_pyg_data()
        torch.save(graph_data, os.path.join(output_dir, 'graph_data.pt'))
        
        # Compute and save statistics
        stats = self._compute_statistics(examples)
        stats.update(graph_stats)
        
        with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  Done! Dataset saved to {output_dir}")
        return stats
    
    def _extract_defects_from_git(
        self,
        repo_path: str
    ) -> Dict[str, List[int]]:
        """
        Extract defect information from git history.
        
        Looks for commits with "fix", "bug", "issue" in message.
        Maps fixed lines to defective labels.
        """
        defects: Dict[str, List[int]] = {}
        
        try:
            # Find bug-fix commits
            result = subprocess.run(
                ['git', 'log', '--grep=fix\\|bug\\|issue', '--oneline', '-n', '500'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"    Warning: Could not read git history")
                return defects
            
            commit_hashes = [line.split()[0] for line in result.stdout.strip().split('\n') if line]
            
            for commit_hash in commit_hashes[:100]:  # Limit to 100 commits
                # Get diff for this commit
                diff_result = subprocess.run(
                    ['git', 'show', '--diff-filter=M', '--name-only', commit_hash],
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )
                
                if diff_result.returncode == 0:
                    # Parse changed files and lines
                    for line in diff_result.stdout.split('\n'):
                        if line.endswith(('.py', '.java', '.c', '.cpp')):
                            # Get detailed diff
                            detailed = subprocess.run(
                                ['git', 'show', f'{commit_hash}^:{line}'],
                                cwd=repo_path,
                                capture_output=True,
                                text=True
                            )
                            
                            if detailed.returncode == 0:
                                # Find functions in this file
                                for func_id, func_info in self.parser.functions.items():
                                    if line in func_id:
                                        # Mark some lines as potentially defective
                                        # (simplified - real impl would parse diff)
                                        if func_id not in defects:
                                            defects[func_id] = []
                                        defects[func_id].extend(
                                            range(func_info.start_line, 
                                                  min(func_info.start_line + 5, func_info.end_line))
                                        )
        
        except Exception as e:
            print(f"    Warning: Error extracting git history: {e}")
        
        return defects
    
    def _create_examples(
        self,
        functions: Dict[str, FunctionInfo],
        defect_labels: Dict[str, List[int]]
    ) -> List[TrainingExample]:
        """Create training examples from functions and labels."""
        examples = []
        example_idx = 0
        
        for func_id, func_info in functions.items():
            defective_lines = set(defect_labels.get(func_id, []))
            
            for rel_line_num, line_code in enumerate(func_info.lines):
                abs_line_num = func_info.start_line + rel_line_num
                
                # Skip empty or comment-only lines
                stripped = line_code.strip()
                if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                    continue
                
                # Get context lines
                context_start = max(0, rel_line_num - self.context_window)
                context_end = min(len(func_info.lines), rel_line_num + self.context_window + 1)
                context_lines = func_info.lines[context_start:context_end]
                
                # Combine context + current line for tokenization
                context_code = '\n'.join(context_lines)
                
                # Tokenize
                tokens = self._tokenize_with_context(
                    context_code,
                    rel_line_num - context_start  # Position of current line in context
                )
                
                # Determine label
                label = 1 if abs_line_num in defective_lines else 0
                
                example = TrainingExample(
                    example_id=f"{func_id}:{abs_line_num}",
                    function_id=func_id,
                    line_number=abs_line_num,
                    line_code=line_code,
                    token_ids=tokens['token_ids'],
                    line_positions=tokens['line_positions'],
                    preceding_mask=tokens['preceding_mask'],
                    attention_mask=tokens['attention_mask'],
                    label=label,
                    context_lines=context_lines,
                    metadata={
                        'module': func_info.module_id,
                        'function_name': func_info.name,
                        'complexity': func_info.complexity
                    }
                )
                
                examples.append(example)
                example_idx += 1
        
        return examples
    
    def _tokenize_with_context(
        self,
        code: str,
        current_line_idx: int
    ) -> Dict[str, List]:
        """
        Tokenize code while tracking line positions.
        """
        lines = code.split('\n')
        
        all_tokens = []
        line_positions = []
        preceding_mask = []
        
        for line_idx, line in enumerate(lines):
            # Tokenize this line
            line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
            
            for token in line_tokens:
                all_tokens.append(token)
                line_positions.append(line_idx)
                preceding_mask.append(line_idx < current_line_idx)
        
        # Truncate to max_length
        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[:self.max_length]
            line_positions = line_positions[:self.max_length]
            preceding_mask = preceding_mask[:self.max_length]
        
        # Create attention mask
        attention_mask = [1] * len(all_tokens)
        
        return {
            'token_ids': all_tokens,
            'line_positions': line_positions,
            'preceding_mask': preceding_mask,
            'attention_mask': attention_mask
        }
    
    def _save_dataset(
        self,
        examples: List[TrainingExample],
        output_dir: str
    ):
        """Save dataset to disk."""
        # Save as JSON for easy inspection
        json_path = os.path.join(output_dir, 'examples.json')
        
        # Convert to dict format
        examples_dict = []
        for ex in examples:
            ex_dict = asdict(ex)
            # Convert numpy arrays if any
            for key, value in ex_dict.items():
                if isinstance(value, np.ndarray):
                    ex_dict[key] = value.tolist()
            examples_dict.append(ex_dict)
        
        with open(json_path, 'w') as f:
            json.dump(examples_dict, f)
        
        # Save in efficient HDF5 format for training
        h5_path = os.path.join(output_dir, 'dataset.h5')
        
        with h5py.File(h5_path, 'w') as f:
            # Create datasets
            max_len = self.max_length
            n = len(examples)
            
            token_ids = f.create_dataset('token_ids', shape=(n, max_len), dtype='i4')
            line_positions = f.create_dataset('line_positions', shape=(n, max_len), dtype='i4')
            preceding_mask = f.create_dataset('preceding_mask', shape=(n, max_len), dtype='bool')
            attention_mask = f.create_dataset('attention_mask', shape=(n, max_len), dtype='i4')
            labels = f.create_dataset('labels', shape=(n,), dtype='i4')
            
            for i, ex in enumerate(examples):
                # Pad to max_length
                def pad(seq, val=0):
                    if len(seq) >= max_len:
                        return seq[:max_len]
                    return seq + [val] * (max_len - len(seq))
                
                token_ids[i] = pad(ex.token_ids)
                line_positions[i] = pad(ex.line_positions)
                preceding_mask[i] = pad(ex.preceding_mask, False)
                attention_mask[i] = pad(ex.attention_mask)
                labels[i] = ex.label
            
            # Save metadata
            f.attrs['num_examples'] = n
            f.attrs['max_length'] = max_len
            f.attrs['created'] = datetime.now().isoformat()
    
    def _compute_statistics(
        self,
        examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Compute dataset statistics."""
        labels = [ex.label for ex in examples]
        num_defective = sum(labels)
        
        return {
            'num_examples': len(examples),
            'num_defective': num_defective,
            'num_clean': len(examples) - num_defective,
            'defect_ratio': num_defective / max(1, len(examples)),
            'avg_token_length': np.mean([len(ex.token_ids) for ex in examples]),
            'processing_date': datetime.now().isoformat()
        }
    
    def load_dataset(
        self,
        data_dir: str,
        split: str = 'train'
    ) -> DefectDataset:
        """
        Load a processed dataset.
        
        Args:
            data_dir: Directory containing processed data
            split: 'train', 'val', or 'test'
        
        Returns:
            DefectDataset ready for DataLoader
        """
        h5_path = os.path.join(data_dir, 'dataset.h5')
        
        examples = []
        
        with h5py.File(h5_path, 'r') as f:
            n = f.attrs['num_examples']
            
            # Determine split indices
            train_end = int(n * 0.8)
            val_end = int(n * 0.9)
            
            if split == 'train':
                indices = range(0, train_end)
            elif split == 'val':
                indices = range(train_end, val_end)
            else:  # test
                indices = range(val_end, n)
            
            for i in indices:
                example = TrainingExample(
                    example_id=f"example_{i}",
                    function_id="",
                    line_number=0,
                    line_code="",
                    token_ids=f['token_ids'][i].tolist(),
                    line_positions=f['line_positions'][i].tolist(),
                    preceding_mask=f['preceding_mask'][i].tolist(),
                    attention_mask=f['attention_mask'][i].tolist(),
                    label=int(f['labels'][i]),
                    context_lines=[],
                    metadata={}
                )
                examples.append(example)
        
        return DefectDataset(examples, self.max_length)
    
    def create_dataloaders(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test DataLoaders.
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        train_dataset = self.load_dataset(data_dir, 'train')
        val_dataset = self.load_dataset(data_dir, 'val')
        test_dataset = self.load_dataset(data_dir, 'test')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
