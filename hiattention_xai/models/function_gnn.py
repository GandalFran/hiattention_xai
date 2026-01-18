"""
Level 3: Function Dependency GNN

Graph Neural Network for modeling function-level dependencies including:
- Call graph edges (function A calls function B)
- Data flow edges (shared variable dependencies)
- Module edges (same module/file membership)

Uses hybrid GCN + GAT architecture for message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class EdgeTypeEmbedding(nn.Module):
    """Embeds different edge types (call, data-flow, module)."""
    
    def __init__(self, num_edge_types: int = 3, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_edge_types, embedding_dim)
        
    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.embedding(edge_type)


class GNNLayer(nn.Module):
    """
    Single GNN layer combining GCN and GAT.
    GCN captures global structure, GAT learns local attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_gat: bool = True
    ):
        super().__init__()
        self.use_gat = use_gat
        
        # GCN for global message passing
        self.gcn = GCNConv(in_channels, out_channels)
        
        # GAT for attention-based aggregation
        if use_gat:
            # GAT output is num_heads * out_channels // num_heads
            self.gat = GATConv(
                in_channels,
                out_channels // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
            self.gat_proj = nn.Linear(out_channels, out_channels)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (if dimensions match)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN layer.
        
        Args:
            x: [num_nodes, in_channels] - Node features
            edge_index: [2, num_edges] - Edge indices (COO format)
            edge_weight: [num_edges] - Optional edge weights
        
        Returns:
            out: [num_nodes, out_channels] - Updated node features
        """
        # Residual
        residual = self.residual(x)
        
        # GCN pass
        gcn_out = self.gcn(x, edge_index, edge_weight)
        
        if self.use_gat:
            # GAT pass
            gat_out = self.gat(x, edge_index)
            gat_out = self.gat_proj(gat_out)
            
            # Combine GCN + GAT
            out = gcn_out + gat_out
        else:
            out = gcn_out
        
        # Batch norm + activation + residual
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = out + residual
        
        return out


class FunctionDependencyGNN(nn.Module):
    """
    Graph Neural Network for modeling function-level dependencies.
    
    Graph structure:
    - Nodes: Function embeddings (from Level 2 encoder)
    - Edges: 
        - Type 0: Call edges (A calls B)
        - Type 1: Data flow edges (A writes var, B reads var)
        - Type 2: Module edges (both in same file/module)
    
    Goal: Propagate information about function behavior through dependency graph
    to capture cross-function context that local analysis misses.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [256, 256, 256],
        output_dim: int = 256,
        num_heads: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.3,
        use_gat: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_edge_types = num_edge_types
        
        # Edge type embedding
        self.edge_embedding = EdgeTypeEmbedding(num_edge_types, 64)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # GNN layers
        dims = hidden_dims + [output_dim]
        self.gnn_layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.gnn_layers.append(
                GNNLayer(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    num_heads=num_heads,
                    dropout=dropout,
                    use_gat=use_gat
                )
            )
        
        # Edge weight computation (based on edge type)
        self.edge_weight_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Graph-level readout
        self.graph_readout = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),  # concat mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        func_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through function dependency GNN.
        
        Args:
            func_embeddings: [num_functions, input_dim] - Node features
            edge_index: [2, num_edges] - Edge indices (COO format)
            edge_attr: [num_edges] - Edge type IDs (0=call, 1=data-flow, 2=module)
            batch: [num_functions] - Batch assignment for each node
        
        Returns:
            node_embeddings: [num_functions, output_dim] - Updated function embeddings
            graph_embedding: [batch_size, output_dim] - Graph-level embedding
        """
        # Project input
        x = self.input_proj(func_embeddings)
        
        # Compute edge weights from edge types
        edge_type_emb = self.edge_embedding(edge_attr)  # [num_edges, 64]
        edge_weight = self.edge_weight_net(edge_type_emb).squeeze(-1)  # [num_edges]
        
        # Message passing through GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_weight)
        
        # Apply layer normalization
        node_embeddings = self.layer_norm(x)
        
        # Graph-level readout
        if batch is None:
            # Single graph case
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Mean + Max pooling
        mean_pool = global_mean_pool(node_embeddings, batch)
        max_pool = global_max_pool(node_embeddings, batch)
        
        graph_features = torch.cat([mean_pool, max_pool], dim=-1)
        graph_embedding = self.graph_readout(graph_features)
        
        return node_embeddings, graph_embedding
    
    def get_node_importance(
        self,
        func_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance score for each function based on graph structure.
        Uses degree centrality and attention weights.
        
        Returns:
            importance_scores: [num_functions] - Importance of each function
        """
        num_nodes = func_embeddings.size(0)
        
        # Compute degree centrality
        in_degrees = torch.zeros(num_nodes, device=func_embeddings.device)
        out_degrees = torch.zeros(num_nodes, device=func_embeddings.device)
        
        src, dst = edge_index
        out_degrees.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        in_degrees.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        
        # Normalize
        total_degree = in_degrees + out_degrees
        importance = total_degree / (total_degree.sum() + 1e-8)
        
        return importance


class CodeGraphBuilder:
    """
    Utility class to build PyTorch Geometric graphs from code repositories.
    Extracts call graphs, data flow, and module structure.
    """
    
    EDGE_TYPE_CALL = 0
    EDGE_TYPE_DATAFLOW = 1
    EDGE_TYPE_MODULE = 2
    
    def __init__(self):
        self.functions: Dict[str, Dict] = {}
        self.call_edges: List[Tuple[str, str]] = []
        self.dataflow_edges: List[Tuple[str, str]] = []
        self.module_map: Dict[str, str] = {}  # func_id -> module_id
    
    def add_function(self, func_id: str, embedding: torch.Tensor, module_id: str):
        """Register a function with its embedding and module."""
        self.functions[func_id] = {
            'embedding': embedding,
            'module': module_id
        }
        self.module_map[func_id] = module_id
    
    def add_call_edge(self, caller_id: str, callee_id: str):
        """Add a call relationship."""
        if caller_id in self.functions and callee_id in self.functions:
            self.call_edges.append((caller_id, callee_id))
    
    def add_dataflow_edge(self, writer_id: str, reader_id: str):
        """Add a data flow dependency."""
        if writer_id in self.functions and reader_id in self.functions:
            self.dataflow_edges.append((writer_id, reader_id))
    
    def build_graph(self) -> Data:
        """
        Build PyTorch Geometric Data object from collected information.
        
        Returns:
            Data object with x, edge_index, edge_attr
        """
        func_ids = list(self.functions.keys())
        id_to_idx = {fid: idx for idx, fid in enumerate(func_ids)}
        
        # Node features
        x = torch.stack([self.functions[fid]['embedding'] for fid in func_ids])
        
        # Build edges
        edges = []
        edge_types = []
        
        # Call edges (type 0)
        for caller, callee in self.call_edges:
            edges.append([id_to_idx[caller], id_to_idx[callee]])
            edge_types.append(self.EDGE_TYPE_CALL)
        
        # Data flow edges (type 1)
        for writer, reader in self.dataflow_edges:
            edges.append([id_to_idx[writer], id_to_idx[reader]])
            edge_types.append(self.EDGE_TYPE_DATAFLOW)
        
        # Module edges (type 2) - connect functions in same module
        modules = {}
        for fid, module in self.module_map.items():
            if module not in modules:
                modules[module] = []
            modules[module].append(fid)
        
        for module_funcs in modules.values():
            for i, f1 in enumerate(module_funcs):
                for f2 in module_funcs[i+1:]:
                    edges.append([id_to_idx[f1], id_to_idx[f2]])
                    edge_types.append(self.EDGE_TYPE_MODULE)
                    # Bidirectional
                    edges.append([id_to_idx[f2], id_to_idx[f1]])
                    edge_types.append(self.EDGE_TYPE_MODULE)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            # Empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.long)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            func_ids=func_ids
        )
    
    def clear(self):
        """Reset builder state."""
        self.functions.clear()
        self.call_edges.clear()
        self.dataflow_edges.clear()
        self.module_map.clear()
