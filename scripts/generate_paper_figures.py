from __future__ import annotations

from pathlib import Path

import numpy as np

import plotly.graph_objects as go

# Matplotlib is used only for the attention heatmap (as requested)
import matplotlib.pyplot as plt

# --- Paths (robust to current working directory) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # proposal-draft/
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotly style aligned with: phd-5-federated-computing/images/generate_figures.py
PLOTLY_TEMPLATE = "plotly_white"


# --- Color system (consistent across figures) ---
# Plotly-friendly colors (similar vibe to the reference script).
COLORS = {
    "primary": "#1f77b4",  # Plotly default blue
    "accent": "#ff7f0e",   # Plotly default orange
    "neutral_dark": "#4D4D4D",
    "neutral_light": "#C7C7C7",
}

# One color per model (consistent across runs/figures)
MODEL_COLORS = {
    "DeepLineDP": "#1f77b4",      # blue
    "PLEASE": "#ff7f0e",          # orange
    "TF-IDF+LR": "#2ca02c",       # green
    "LineVul": "#d62728",         # red
    "HiAttention-XAI": "#9467bd", # purple
}


def _write_outputs(fig: go.Figure, stem: str, *, width: int, height: int) -> None:
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    html_path = OUTPUT_DIR / f"{stem}.html"
    fig.write_image(str(pdf_path), width=width, height=height)
    fig.write_html(str(html_path))


def _save_matplotlib_pdf(filename: str, *, dpi: int = 300) -> None:
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)


# --- GRÁFICO 1: COMPARACIÓN ---
models = ['DeepLineDP', 'PLEASE', 'TF-IDF+LR', 'LineVul', 'HiAttention-XAI']
f1_scores = [0.380, 0.450, 0.875, 0.930, 0.945]

bar_colors = [MODEL_COLORS.get(m, COLORS["neutral_light"]) for m in models]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=models,
        y=f1_scores,
        marker_color=bar_colors,
        marker_line=dict(color=COLORS["neutral_dark"], width=1),
        text=[f"{v:.3f}" for v in f1_scores],
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
    )
)

fig.update_layout(
    title="Performance Comparison with State-of-the-Art",
    xaxis_title="",
    yaxis_title="F1-Score",
    template=PLOTLY_TEMPLATE,
    height=500,
    width=850,
    margin=dict(l=70, r=30, t=70, b=70),
)
fig.update_yaxes(range=[0, 1.08], gridcolor="#D9D9D9")
fig.update_xaxes(tickangle=-15)

_write_outputs(fig, "comparison_chart", width=850, height=500)
print("Generado comparison_chart.pdf")

# --- GRÁFICO 2: ABLACIÓN ---
variants = ['L2 Only (CodeT5)', 'L2 + L3 (Graph)', 'Full Model (L2+L3+L4)']
ablation_scores = [0.915, 0.932, 0.945] # Valores hipotéticos coherentes con tu historia

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=variants,
        y=ablation_scores,
        mode="lines+markers+text",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=10, color=COLORS["primary"], line=dict(width=1, color="white")),
        text=[f"{v:.3f}" for v in ablation_scores],
        textposition="top center",
        textfont=dict(color=COLORS["neutral_dark"], size=12),
        showlegend=False,
    )
)

fig.update_layout(
    title="Ablation Study: Impact of Hierarchical Context",
    xaxis_title="",
    yaxis_title="F1-Score",
    template=PLOTLY_TEMPLATE,
    height=500,
    width=850,
    margin=dict(l=70, r=30, t=70, b=90),
)
fig.update_yaxes(range=[0.90, 0.96], gridcolor="#D9D9D9")

_write_outputs(fig, "ablation_chart", width=850, height=500)
print("Generado ablation_chart.pdf")

# --- GRÁFICO 3: ATTENTION HEATMAP (XAI CASE STUDY) ---
def plot_attention_heatmap():
    
    # Sample Vulnerable Code (SQL Injection) - Split for clarity/width
    code_lines = [
        "String user = request.getParameter(\"user\");",
        "String query = \"SELECT * FROM users\"",
        "             + \" WHERE name = '\" + user + \"'\";",
        "Statement stmt = conn.createStatement();",
        "ResultSet rs = stmt.executeQuery(query);",
        "// Vulnerability: Unsanitized input concatenation"
    ]
    
    # Synthetic Attention Weights
    attention_scores = np.array([
        [0.05],  # getParameter
        [0.10],  # SELECT part
        [0.85],  # WHERE part + concatenation (CRITICAL)
        [0.05],  # createStatement
        [0.05],  # executeQuery
        [0.00]   # comment
    ])

    # Keep the original Matplotlib-based look (as requested)
    try:
        plt.style.use('seaborn-v0_8-paper')
    except Exception:
        try:
            plt.style.use('seaborn-paper')
        except Exception:
            pass

    plt.figure(figsize=(12, 4))  # Wider to ensure text fits

    # Plot using standard Reds colormap
    plt.imshow(attention_scores, aspect='auto', cmap='Reds', vmin=0, vmax=1)

    # Add text
    for i in range(len(code_lines)):
        score = attention_scores[i][0]
        # Text color: White on dark red (>0.5), Black on light
        text_color = 'white' if score > 0.5 else 'black'

        # Coordinates: x=-0.45 (left edge), i (y row)
        plt.text(
            -0.45,
            i,
            f"{code_lines[i]}",
            ha='left',
            va='center',
            fontsize=12,
            family='monospace',
            color=text_color,
        )

    plt.axis('off')  # Hide axes
    plt.title('Case Study: XAI Attention Heatmap on Vulnerable Code', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    _save_matplotlib_pdf('attention_heatmap.pdf', dpi=300)
    plt.close()
    print("Generado attention_heatmap.pdf")

if __name__ == "__main__":
    plot_attention_heatmap()
