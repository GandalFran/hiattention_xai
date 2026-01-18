"""
Level 2: Local Context Encoder

Encodes line-level semantic information with local dependencies using:
- CodeT5 for token embedding
- BiLSTM for sequential context
- Multi-head self-attention
- Preceding Line Aware module (from PLEASE paper)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModel, AutoTokenizer


class PrecedingLineAwareModule(nn.Module):
    """
    Enhances embeddings by emphasizing how preceding lines influence current line.
    Based on PLEASE framework insight that preceding lines carry critical context.
    """
    
    def __init__(self, embedding_dim: int, context_window: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        
        # Learnable distance weights for preceding lines
        self.distance_weights = nn.Parameter(
            torch.ones(context_window) / context_window
        )
        
        # GRU for modeling sequential influence from preceding lines
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention for weighting preceding influence
        self.attention = nn.Linear(embedding_dim, 1)
        
        # Fusion layer
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        
    def forward(
        self,
        current_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
        preceding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhance current line embeddings based on preceding line impact.
        
        Args:
            current_embeddings: [batch, seq_len, embedding_dim]
            context_embeddings: [batch, seq_len, embedding_dim] from BiLSTM
            preceding_mask: [batch, seq_len] boolean mask for preceding tokens
        
        Returns:
            enhanced_embeddings: [batch, seq_len, embedding_dim]
        """
        batch_size, seq_len, emb_dim = current_embeddings.shape
        
        # Apply mask to isolate preceding line tokens
        masked_context = context_embeddings * preceding_mask.unsqueeze(-1).float()
        
        # Process through GRU to capture sequential dependencies
        gru_out, hidden = self.gru(masked_context)
        
        # Compute attention weights for preceding tokens
        attn_scores = self.attention(gru_out).squeeze(-1)  # [batch, seq_len]
        attn_scores = attn_scores.masked_fill(~preceding_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Weighted sum of preceding context
        preceding_context = torch.bmm(
            attn_weights.unsqueeze(1),
            gru_out
        ).squeeze(1)  # [batch, embedding_dim]
        
        # Expand to match sequence length
        preceding_context = preceding_context.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Fuse with current embeddings
        combined = torch.cat([current_embeddings, preceding_context], dim=-1)
        enhanced = self.fusion(combined)
        
        return enhanced


class LocalContextEncoder(nn.Module):
    """
    Level 2 encoder for local semantic context.
    
    Architecture:
    1. CodeT5 token embeddings (pretrained)
    2. Position encoding
    3. BiLSTM for sequential modeling
    4. Multi-head self-attention
    5. Preceding Line Aware enhancement
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        context_window: int = 5,
        dropout: float = 0.3,
        pretrained_model: str = "Salesforce/codet5-base",
        use_pretrained: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embedding layer
        if use_pretrained:
            # Load CodeT5 encoder for embeddings
            self.pretrained = AutoModel.from_pretrained(pretrained_model)
            self.pretrained_dim = self.pretrained.config.d_model
            # Project to our embedding dim
            self.embedding_proj = nn.Linear(self.pretrained_dim, embedding_dim)
            self.token_embedding = None
        else:
            self.pretrained = None
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding_proj = None
        
        # Position encoding (absolute positions within file)
        self.position_embedding = nn.Embedding(4096, embedding_dim)  # Max 4096 positions
        
        # Line position encoding (which line number)
        self.line_embedding = nn.Embedding(1000, embedding_dim)  # Max 1000 lines
        
        # BiLSTM for sequential context
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Project BiLSTM output back to embedding_dim
        self.lstm_proj = nn.Linear(hidden_dim * 2, embedding_dim)
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Preceding Line Aware Module
        self.preceding_aware = PrecedingLineAwareModule(
            embedding_dim=embedding_dim,
            context_window=context_window
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
    def forward(
        self,
        token_ids: torch.Tensor,
        line_positions: torch.Tensor,
        preceding_lines_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through local context encoder.
        
        Args:
            token_ids: [batch, seq_len] - Token indices
            line_positions: [batch, seq_len] - Line number for each token
            preceding_lines_mask: [batch, seq_len] - Mask for preceding line tokens
            attention_mask: [batch, seq_len] - Padding mask
        
        Returns:
            embeddings: [batch, seq_len, embedding_dim] - Contextual embeddings
            attention_weights: [batch, num_heads, seq_len, seq_len] - Attention patterns
        """
        batch_size, seq_len = token_ids.shape
        
        # 1. Token embeddings
        if self.pretrained is not None:
            with torch.no_grad():
                pretrained_out = self.pretrained(
                    input_ids=token_ids,
                    attention_mask=attention_mask
                ).last_hidden_state
            token_emb = self.embedding_proj(pretrained_out)
        else:
            token_emb = self.token_embedding(token_ids)
        
        # 2. Position encoding
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_emb = self.position_embedding(positions.clamp(max=4095))
        
        # Line-level position
        line_emb = self.line_embedding(line_positions.clamp(max=999))
        
        # Combine embeddings
        x = token_emb + pos_emb + line_emb
        x = self.dropout(x)
        
        # 3. BiLSTM
        if attention_mask is not None:
            # Pack padded sequence for efficiency
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm_out, _ = self.bilstm(x)
        
        lstm_out = self.lstm_proj(lstm_out)
        
        # 4. Self-attention with residual
        attn_mask = None
        if attention_mask is not None:
            # Create attention mask (True means ignore)
            attn_mask = ~attention_mask.bool()
        
        attn_out, attn_weights = self.self_attention(
            x, x, x,
            key_padding_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        x = self.norm1(x + attn_out)
        
        # 5. Preceding Line Aware enhancement
        preceding_enhanced = self.preceding_aware(
            x, lstm_out, preceding_lines_mask
        )
        x = self.norm2(x + preceding_enhanced)
        
        # 6. Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm3(x + ff_out)
        
        return x, attn_weights


class LocalContextTokenizer:
    """
    Tokenizer wrapper for code with line-level information.
    """
    
    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def tokenize_with_lines(
        self,
        code: str,
        max_length: int = 512,
        context_window: int = 5
    ) -> dict:
        """
        Tokenize code while preserving line information.
        
        Returns:
            dict with token_ids, line_positions, preceding_mask, attention_mask
        """
        lines = code.split('\n')
        
        all_tokens = []
        line_positions = []
        
        for line_idx, line in enumerate(lines):
            line_tokens = self.tokenizer.encode(
                line, add_special_tokens=False
            )
            all_tokens.extend(line_tokens)
            line_positions.extend([line_idx] * len(line_tokens))
        
        # Truncate to max_length
        if len(all_tokens) > max_length:
            all_tokens = all_tokens[:max_length]
            line_positions = line_positions[:max_length]
        
        # Pad to max_length
        padding_length = max_length - len(all_tokens)
        attention_mask = [1] * len(all_tokens) + [0] * padding_length
        all_tokens = all_tokens + [self.tokenizer.pad_token_id] * padding_length
        line_positions = line_positions + [0] * padding_length
        
        # Create preceding line mask (tokens from lines before current)
        preceding_mask = []
        current_line = line_positions[0] if line_positions else 0
        for pos in line_positions[:len(attention_mask) - padding_length]:
            if pos < current_line:
                preceding_mask.append(True)
            else:
                preceding_mask.append(False)
                current_line = pos
        preceding_mask.extend([False] * padding_length)
        
        return {
            'token_ids': torch.tensor(all_tokens),
            'line_positions': torch.tensor(line_positions),
            'preceding_mask': torch.tensor(preceding_mask),
            'attention_mask': torch.tensor(attention_mask)
        }
