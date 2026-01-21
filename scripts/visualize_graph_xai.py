
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Import model definition
sys.path.append(str(Path(__file__).parent))
from train_kg_xai import KG_XAI_Model, ExpertFeatureExtractor
from hiattention_xai.data.simple_graph_builder import SimpleCodeGraphBuilder

def visualize_attention(model_path, code_sample, output_path='graph_xai.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = KG_XAI_Model()
    if Path(model_path).exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            print("Warning: Could not load weights perfectly, strict=False")
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        print(f"Warning: Model path {model_path} not found. Using initialized weights for DEMO.")

    
    model.to(device)
    model.eval()
    
    # 2. Hook to capture attention
    attention_weights = {}
    def get_attention(name):
        def hook(module, input, output):
            # output of GATConv with return_attention_weights=True is (out, (edge_index, alpha))
            # But standard forward returns just out. We need to force return_attention_weights in the model or access internal alpha if stored.
            # PyG GATConv doesn't store alpha by default.
            pass
        return hook

    # Since standard GATConv forward doesn't return attention unless specified, 
    # and the training model didn't specify it, we might need to monkey-patch or use a wrapper.
    # A cleaner hack for XAI on an existing model:
    # We can replace the forward method of the specific GATConv layers instance for this inference.
    
    encoded_graph = None
    captured_alpha = []
    
    def forward_with_attn(self, x, edge_index, size=None, return_attention_weights=True):
        # We force return_attention_weights=True
        return super(type(self), self).forward(x, edge_index, size=size, return_attention_weights=return_attention_weights)

    # 3. Build Graph
    builder = SimpleCodeGraphBuilder()
    x, edge_index, edge_attr = builder.to_pytorch_geometric(code_sample)
    graph_data = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)]).to(device)
    
    # 4. Prepare Inputs
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
    encoding = tokenizer(code_sample, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    mask = encoding['attention_mask'].to(device)
    expert = torch.tensor([ExpertFeatureExtractor.extract_features(code_sample)], dtype=torch.float32).to(device)

    # 5. Run Inference with Attention Capture
    # We will manually run the graph encoder part to get attention
    # We access the conv1 layer directly
    
    with torch.no_grad():
        # Run GATConv 1 manually to get attention
        x_in = graph_data.x
        edge_index_in = graph_data.edge_index
        
        # Call the layer but force it to return attention
        # Note: This relies on PyG implementation. 
        # For GATConv, if we pass return_attention_weights=True, it returns (out, (edge_index, alpha))
        
        out1, (idx1, alpha1) = model.graph_encoder.conv1(x_in, edge_index_in, return_attention_weights=True)
        # alpha1 is [num_edges, num_heads]
        
        # Average attention across heads
        att_weights = alpha1.mean(dim=1).cpu().numpy()
        
    # 6. Plot
    lines = code_sample.split('\n')
    G = nx.DiGraph()
    
    # Add nodes (lines)
    for i, line in enumerate(lines):
        G.add_node(i, label=f"{i+1}: {line[:30]}...")
        
    # Add edges with attention
    edges = idx1.cpu().numpy()
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        w = att_weights[i]
        G.add_edge(src, dst, weight=w)
        
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    
    # Draw edges with color based on attention
    edges_list = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges_list]
    
    # Normalize weights for visualization
    if len(weights) > 0:
        max_w = max(weights)
        weights = [w/max_w for w in weights]
        
    nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Reds, width=2, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("KG-XAI Attention Visualization (Red = High Symbolic Importance)")
    plt.axis('off')
    plt.tight_layout()
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Graph XAI saved to {output_path}")

if __name__ == "__main__":
    # Demo sample: A slightly more complex buffer overflow pattern with control flow
    sample_code = """
void process_user_data(char *input_str) {
    int len = strlen(input_str);
    char buffer[64];
    
    // Safety check - looks structurally fine but might be bypassed
    if (len < 100) { 
        // Flaw: 100 is still larger than buffer size (64)
        strcpy(buffer, input_str); 
        printf("Processed: %s", buffer);
    } else {
        printf("Input too long!");
    }
}
    """
    
    # Ensure figures directory exists
    Path('paper/figures').mkdir(parents=True, exist_ok=True)
    
    # Run visualization
    visualize_attention('results/kg_xai_fusion_model.pt', sample_code, output_path='paper/figures/attention_heatmap.pdf')
