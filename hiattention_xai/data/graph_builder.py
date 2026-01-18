"""
Graph Builder Module

Constructs call graphs and data flow graphs from parsed code.
Creates PyTorch Geometric compatible graph structures.
"""

import torch
from torch_geometric.data import Data
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import networkx as nx

from .code_parser import FunctionInfo, ModuleInfo, CodeParser


@dataclass
class CallGraphEdge:
    """Represents a call relationship between functions."""
    caller_id: str
    callee_id: str
    call_site_line: int = -1
    is_direct: bool = True


@dataclass
class DataFlowEdge:
    """Represents a data flow relationship (shared variable)."""
    writer_id: str
    reader_id: str
    variable_name: str


class CodeGraphBuilder:
    """
    Builds graph representations of code repositories.
    
    Creates:
    1. Call graph (function A calls function B)
    2. Data flow graph (function A writes variable X, function B reads it)
    3. Module graph (functions in same module)
    
    Outputs PyTorch Geometric Data objects for GNN processing.
    """
    
    # Edge type constants
    EDGE_CALL = 0
    EDGE_DATAFLOW = 1
    EDGE_MODULE = 2
    
    def __init__(self):
        self.functions: Dict[str, FunctionInfo] = {}
        self.modules: Dict[str, ModuleInfo] = {}
        self.call_edges: List[CallGraphEdge] = []
        self.dataflow_edges: List[DataFlowEdge] = []
        
        # Mappings for quick lookup
        self.func_id_to_idx: Dict[str, int] = {}
        self.module_id_to_idx: Dict[str, int] = {}
        
        # NetworkX graphs for analysis
        self.nx_call_graph: Optional[nx.DiGraph] = None
        self.nx_module_graph: Optional[nx.DiGraph] = None
    
    def build_from_parser(self, parser: CodeParser):
        """
        Build graphs from a CodeParser instance.
        
        Args:
            parser: Parsed code repository
        """
        self.functions = parser.functions
        self.modules = parser.modules
        
        # Build ID mappings
        self.func_id_to_idx = {fid: idx for idx, fid in enumerate(self.functions.keys())}
        self.module_id_to_idx = {mid: idx for idx, mid in enumerate(self.modules.keys())}
        
        # Build call graph
        self._build_call_graph()
        
        # Build data flow graph
        self._build_dataflow_graph()
        
        # Build NetworkX graphs
        self._build_networkx_graphs()
    
    def _build_call_graph(self):
        """Extract call relationships from function information."""
        self.call_edges = []
        
        # Build name to ID mapping for call resolution
        name_to_ids: Dict[str, List[str]] = {}
        for func_id, func_info in self.functions.items():
            name = func_info.name
            if name not in name_to_ids:
                name_to_ids[name] = []
            name_to_ids[name].append(func_id)
        
        # Find call edges
        for caller_id, caller_info in self.functions.items():
            caller_module = caller_info.module_id
            
            for callee_name in caller_info.calls:
                # Try to resolve callee
                callee_candidates = name_to_ids.get(callee_name, [])
                
                if not callee_candidates:
                    continue
                
                # Prefer same module
                same_module = [c for c in callee_candidates 
                              if self.functions[c].module_id == caller_module]
                
                if same_module:
                    callee_id = same_module[0]
                else:
                    callee_id = callee_candidates[0]
                
                if callee_id != caller_id:  # No self-loops
                    self.call_edges.append(CallGraphEdge(
                        caller_id=caller_id,
                        callee_id=callee_id
                    ))
    
    def _build_dataflow_graph(self):
        """Extract data flow relationships from shared variables."""
        self.dataflow_edges = []
        
        # Group functions by module for local variable analysis
        for module_id, module_info in self.modules.items():
            module_funcs = [self.functions[fid] for fid in module_info.functions
                          if fid in self.functions]
            
            # Track which functions write/read which variables
            var_writers: Dict[str, List[str]] = {}
            var_readers: Dict[str, List[str]] = {}
            
            for func_info in module_funcs:
                for var in func_info.variables_written:
                    if var not in var_writers:
                        var_writers[var] = []
                    var_writers[var].append(func_info.func_id)
                
                for var in func_info.variables_read:
                    if var not in var_readers:
                        var_readers[var] = []
                    var_readers[var].append(func_info.func_id)
            
            # Create edges for shared variables
            for var_name in var_writers:
                if var_name in var_readers:
                    for writer_id in var_writers[var_name]:
                        for reader_id in var_readers[var_name]:
                            if writer_id != reader_id:
                                self.dataflow_edges.append(DataFlowEdge(
                                    writer_id=writer_id,
                                    reader_id=reader_id,
                                    variable_name=var_name
                                ))
    
    def _build_networkx_graphs(self):
        """Build NetworkX graphs for analysis."""
        # Call graph
        self.nx_call_graph = nx.DiGraph()
        self.nx_call_graph.add_nodes_from(self.functions.keys())
        
        for edge in self.call_edges:
            self.nx_call_graph.add_edge(edge.caller_id, edge.callee_id)
        
        # Module graph
        self.nx_module_graph = nx.DiGraph()
        self.nx_module_graph.add_nodes_from(self.modules.keys())
        
        # Add edges between modules that have function calls
        for edge in self.call_edges:
            caller_module = self.functions[edge.caller_id].module_id
            callee_module = self.functions[edge.callee_id].module_id
            
            if caller_module != callee_module:
                if self.nx_module_graph.has_edge(caller_module, callee_module):
                    self.nx_module_graph[caller_module][callee_module]['weight'] += 1
                else:
                    self.nx_module_graph.add_edge(caller_module, callee_module, weight=1)
    
    def to_pyg_data(
        self,
        func_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        embedding_dim: int = 256
    ) -> Data:
        """
        Convert to PyTorch Geometric Data object.
        
        Args:
            func_embeddings: Pre-computed embeddings for each function
            embedding_dim: Dimension of embeddings (if generating dummy)
        
        Returns:
            PyTorch Geometric Data object
        """
        num_nodes = len(self.functions)
        func_ids = list(self.functions.keys())
        
        # Node features
        if func_embeddings is not None:
            x = torch.stack([func_embeddings[fid] for fid in func_ids])
        else:
            # Generate dummy embeddings (to be replaced by actual embeddings)
            x = torch.randn(num_nodes, embedding_dim)
        
        # Build edge lists
        edges = []
        edge_types = []
        
        # Call edges (type 0)
        for edge in self.call_edges:
            if edge.caller_id in self.func_id_to_idx and edge.callee_id in self.func_id_to_idx:
                src = self.func_id_to_idx[edge.caller_id]
                dst = self.func_id_to_idx[edge.callee_id]
                edges.append([src, dst])
                edge_types.append(self.EDGE_CALL)
        
        # Data flow edges (type 1)
        seen_dataflow = set()
        for edge in self.dataflow_edges:
            if edge.writer_id in self.func_id_to_idx and edge.reader_id in self.func_id_to_idx:
                src = self.func_id_to_idx[edge.writer_id]
                dst = self.func_id_to_idx[edge.reader_id]
                edge_key = (src, dst, self.EDGE_DATAFLOW)
                if edge_key not in seen_dataflow:
                    edges.append([src, dst])
                    edge_types.append(self.EDGE_DATAFLOW)
                    seen_dataflow.add(edge_key)
        
        # Module edges (type 2) - connect functions in same module
        for module_id, module_info in self.modules.items():
            module_func_ids = [fid for fid in module_info.functions if fid in self.func_id_to_idx]
            
            for i, fid1 in enumerate(module_func_ids):
                for fid2 in module_func_ids[i+1:]:
                    src = self.func_id_to_idx[fid1]
                    dst = self.func_id_to_idx[fid2]
                    # Bidirectional module edges
                    edges.append([src, dst])
                    edge_types.append(self.EDGE_MODULE)
                    edges.append([dst, src])
                    edge_types.append(self.EDGE_MODULE)
        
        # Convert to tensors
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.long)
        
        # Create function to module mapping
        func_to_module = torch.tensor([
            self.module_id_to_idx.get(self.functions[fid].module_id, 0)
            for fid in func_ids
        ], dtype=torch.long)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            func_ids=func_ids,
            func_to_module=func_to_module,
            num_nodes=num_nodes
        )
    
    def get_function_neighbors(
        self,
        func_id: str,
        hop: int = 1,
        edge_types: Optional[List[int]] = None
    ) -> List[str]:
        """
        Get neighboring functions within k hops.
        
        Args:
            func_id: Function to find neighbors for
            hop: Number of hops (1 = direct neighbors)
            edge_types: Filter by edge types (None = all)
        
        Returns:
            List of neighbor function IDs
        """
        if func_id not in self.func_id_to_idx:
            return []
        
        neighbors = set()
        current_level = {func_id}
        
        for _ in range(hop):
            next_level = set()
            
            for fid in current_level:
                # Check call edges
                if edge_types is None or self.EDGE_CALL in edge_types:
                    for edge in self.call_edges:
                        if edge.caller_id == fid:
                            next_level.add(edge.callee_id)
                        if edge.callee_id == fid:
                            next_level.add(edge.caller_id)
                
                # Check dataflow edges
                if edge_types is None or self.EDGE_DATAFLOW in edge_types:
                    for edge in self.dataflow_edges:
                        if edge.writer_id == fid:
                            next_level.add(edge.reader_id)
                        if edge.reader_id == fid:
                            next_level.add(edge.writer_id)
            
            neighbors.update(next_level)
            current_level = next_level - neighbors
        
        neighbors.discard(func_id)
        return list(neighbors)
    
    def compute_centrality(self) -> Dict[str, float]:
        """Compute PageRank centrality for functions."""
        if self.nx_call_graph is None:
            return {}
        
        try:
            return nx.pagerank(self.nx_call_graph)
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree centrality
            return nx.degree_centrality(self.nx_call_graph)
    
    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        return {
            'num_functions': len(self.functions),
            'num_modules': len(self.modules),
            'num_call_edges': len(self.call_edges),
            'num_dataflow_edges': len(self.dataflow_edges),
            'avg_out_degree': len(self.call_edges) / max(1, len(self.functions)),
            'graph_density': len(self.call_edges) / max(1, len(self.functions) ** 2)
        }
    
    def clear(self):
        """Reset builder state."""
        self.functions.clear()
        self.modules.clear()
        self.call_edges.clear()
        self.dataflow_edges.clear()
        self.func_id_to_idx.clear()
        self.module_id_to_idx.clear()
        self.nx_call_graph = None
        self.nx_module_graph = None
