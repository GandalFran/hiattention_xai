"""
Level 4: Architectural Context Analyzer

Analyzes module-level and architectural patterns including:
- Modularity (Newman's algorithm)
- Coupling (afferent/efferent dependencies)
- Cohesion (LCOM - Lack of Cohesion of Methods)
- Technical debt indicators

These architectural metrics provide context that line-level and function-level
analysis cannot capture, such as design patterns and SOLID violations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np


class ModularityEncoder(nn.Module):
    """Encodes repository modularity into embedding."""
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, modularity_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modularity_score: [batch] or [1] - Newman's modularity value
        
        Returns:
            embedding: [batch, embedding_dim]
        """
        if modularity_score.dim() == 1:
            modularity_score = modularity_score.unsqueeze(-1)
        return self.mlp(modularity_score)


class CouplingEncoder(nn.Module):
    """Encodes coupling metrics (afferent and efferent) into embedding."""
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),  # afferent + efferent
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, coupling_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coupling_scores: [num_modules, 2] - (afferent, efferent) per module
        
        Returns:
            embedding: [num_modules, embedding_dim]
        """
        return self.mlp(coupling_scores)


class CohesionEncoder(nn.Module):
    """Encodes cohesion (LCOM) metric into embedding."""
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, cohesion_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cohesion_scores: [num_modules, 1] - LCOM per module
        
        Returns:
            embedding: [num_modules, embedding_dim]
        """
        if cohesion_scores.dim() == 1:
            cohesion_scores = cohesion_scores.unsqueeze(-1)
        return self.mlp(cohesion_scores)


class TechnicalDebtEncoder(nn.Module):
    """
    Encodes technical debt indicators:
    1. Cyclomatic complexity (avg)
    2. Code duplication ratio
    3. Long method count ratio
    4. Deep nesting ratio
    5. SOLID violation count
    """
    
    def __init__(self, num_indicators: int = 5, embedding_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_indicators, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, debt_indicators: torch.Tensor) -> torch.Tensor:
        """
        Args:
            debt_indicators: [num_modules, num_indicators]
        
        Returns:
            embedding: [num_modules, embedding_dim]
        """
        return self.mlp(debt_indicators)


class ArchitecturalContextLayer(nn.Module):
    """
    Analyzes module-level and architectural patterns.
    
    Combines multiple architectural metrics into a unified representation
    that captures design-level context for defect prediction.
    
    This is Level 4 of the hierarchical architecture.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_debt_indicators: int = 5,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Metric encoders
        self.modularity_encoder = ModularityEncoder(embedding_dim)
        self.coupling_encoder = CouplingEncoder(embedding_dim)
        self.cohesion_encoder = CohesionEncoder(embedding_dim)
        self.debt_encoder = TechnicalDebtEncoder(num_debt_indicators, embedding_dim)
        
        # Cross-metric attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Function-level projection
        self.func_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(
        self,
        modularity_score: torch.Tensor,
        coupling_scores: torch.Tensor,
        cohesion_scores: torch.Tensor,
        debt_indicators: torch.Tensor,
        function_to_module: torch.Tensor,
        num_functions: int
    ) -> torch.Tensor:
        """
        Compute architectural context embedding for each function.
        
        Args:
            modularity_score: [1] - Repository modularity
            coupling_scores: [num_modules, 2] - (afferent, efferent)
            cohesion_scores: [num_modules] - LCOM per module
            debt_indicators: [num_modules, 5] - Debt indicators per module
            function_to_module: [num_functions] - Module index for each function
            num_functions: Total number of functions
        
        Returns:
            arch_context: [num_functions, embedding_dim]
        """
        num_modules = coupling_scores.size(0)
        
        # Encode each metric
        mod_emb = self.modularity_encoder(modularity_score)  # [1, E]
        coup_emb = self.coupling_encoder(coupling_scores)    # [M, E]
        coh_emb = self.cohesion_encoder(cohesion_scores)     # [M, E]
        debt_emb = self.debt_encoder(debt_indicators)        # [M, E]
        
        # Expand modularity to all modules
        mod_emb = mod_emb.expand(num_modules, -1)  # [M, E]
        
        # Stack for cross-attention
        # Shape: [num_modules, 4, embedding_dim]
        metric_stack = torch.stack([mod_emb, coup_emb, coh_emb, debt_emb], dim=1)
        
        # Cross-metric attention (let metrics attend to each other)
        attended, _ = self.cross_attention(
            metric_stack, metric_stack, metric_stack
        )
        
        # Concatenate all metric embeddings
        # Shape: [num_modules, 4 * embedding_dim]
        concatenated = attended.view(num_modules, -1)
        
        # Fuse into single embedding per module
        module_context = self.fusion(concatenated)  # [M, E]
        module_context = self.layer_norm(module_context)
        
        # Map module context to function level
        arch_context = module_context[function_to_module]  # [F, E]
        arch_context = self.func_proj(arch_context)
        
        return arch_context


class ArchitecturalMetricsComputer:
    """
    Computes architectural metrics from a NetworkX graph representation
    of the code repository.
    """
    
    @staticmethod
    def compute_modularity(graph: nx.Graph) -> float:
        """
        Compute Newman's modularity using community detection.
        
        Args:
            graph: NetworkX graph where nodes are modules/classes
        
        Returns:
            modularity: Float in [-0.5, 1.0], higher is better
        """
        if graph.number_of_nodes() < 2:
            return 0.0
        
        try:
            # Find communities using greedy modularity optimization
            communities = nx.community.greedy_modularity_communities(graph)
            modularity = nx.community.modularity(graph, communities)
        except Exception:
            modularity = 0.0
        
        return modularity
    
    @staticmethod
    def compute_coupling(graph: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute afferent (incoming) and efferent (outgoing) coupling.
        
        Ca (afferent): Number of classes outside the module that depend on it
        Ce (efferent): Number of classes inside the module that depend on outside
        
        Returns:
            (afferent_coupling, efferent_coupling): Arrays of shape [num_nodes]
        """
        nodes = list(graph.nodes())
        afferent = np.array([graph.in_degree(n) for n in nodes], dtype=np.float32)
        efferent = np.array([graph.out_degree(n) for n in nodes], dtype=np.float32)
        
        # Normalize by max
        if afferent.max() > 0:
            afferent /= afferent.max()
        if efferent.max() > 0:
            efferent /= efferent.max()
        
        return afferent, efferent
    
    @staticmethod
    def compute_lcom(module_methods: Dict[str, List[str]], 
                     shared_variables: Dict[str, List[str]]) -> float:
        """
        Compute Lack of Cohesion of Methods (LCOM).
        
        LCOM = (pairs_not_sharing - pairs_sharing) / total_pairs
        Higher LCOM = lower cohesion = potential design issue
        
        Args:
            module_methods: Dict mapping method_id to list of accessed variable names
            shared_variables: Dict mapping variable_id to list of method_ids that access it
        
        Returns:
            lcom: Float in [0, 1], lower is better
        """
        methods = list(module_methods.keys())
        n = len(methods)
        
        if n < 2:
            return 0.0
        
        pairs_sharing = 0
        pairs_not_sharing = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                vars_i = set(module_methods[methods[i]])
                vars_j = set(module_methods[methods[j]])
                
                if vars_i & vars_j:  # Shared variables
                    pairs_sharing += 1
                else:
                    pairs_not_sharing += 1
        
        total_pairs = n * (n - 1) / 2
        lcom = max(0, (pairs_not_sharing - pairs_sharing) / total_pairs)
        
        return lcom
    
    @staticmethod
    def compute_debt_indicators(
        cyclomatic_complexity: np.ndarray,
        duplication_ratio: np.ndarray,
        long_method_ratio: np.ndarray,
        deep_nesting_ratio: np.ndarray,
        solid_violations: np.ndarray
    ) -> np.ndarray:
        """
        Combine all technical debt indicators into a single array.
        
        All inputs should be arrays of shape [num_modules].
        Values should be normalized to [0, 1] range.
        
        Returns:
            indicators: Array of shape [num_modules, 5]
        """
        return np.stack([
            cyclomatic_complexity,
            duplication_ratio,
            long_method_ratio,
            deep_nesting_ratio,
            solid_violations
        ], axis=1)
    
    @staticmethod
    def compute_all_metrics(
        repo_graph: nx.DiGraph,
        module_methods: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all architectural metrics for a repository.
        
        Args:
            repo_graph: Directed graph of module dependencies
            module_methods: Dict[module] -> Dict[method] -> [variables]
        
        Returns:
            Dict with all computed metrics as tensors
        """
        # Modularity
        undirected = repo_graph.to_undirected()
        modularity = ArchitecturalMetricsComputer.compute_modularity(undirected)
        
        # Coupling
        afferent, efferent = ArchitecturalMetricsComputer.compute_coupling(repo_graph)
        coupling = np.stack([afferent, efferent], axis=1)
        
        # Cohesion per module
        lcom_scores = []
        for module, methods in module_methods.items():
            # Build shared variables map
            shared_vars = {}
            for method, vars_accessed in methods.items():
                for var in vars_accessed:
                    if var not in shared_vars:
                        shared_vars[var] = []
                    shared_vars[var].append(method)
            
            lcom = ArchitecturalMetricsComputer.compute_lcom(methods, shared_vars)
            lcom_scores.append(lcom)
        
        if not lcom_scores:
            lcom_scores = [0.0]
        
        # Placeholder debt indicators (would need actual code analysis)
        num_modules = len(module_methods) if module_methods else 1
        debt_indicators = np.random.rand(num_modules, 5).astype(np.float32) * 0.5  # Placeholder
        
        return {
            'modularity': torch.tensor([modularity], dtype=torch.float32),
            'coupling': torch.tensor(coupling, dtype=torch.float32),
            'cohesion': torch.tensor(lcom_scores, dtype=torch.float32),
            'debt_indicators': torch.tensor(debt_indicators, dtype=torch.float32)
        }
