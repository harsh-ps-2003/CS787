import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadSelfAttention(nn.Module):
    """Implements the Multi-Head Self-Attention mechanism."""
    def __init__(self, d_model, n_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(context)

class TransformerEncoderLayer(nn.Module):
    """A single layer of the Transformer Encoder."""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

class Pooling(nn.Module):
    """Performs pooling over the sequence dimension to get a single vector."""
    def __init__(self, strategy='cls'):
        super(Pooling, self).__init__()
        self.strategy = strategy

    def forward(self, x, mask):
        if self.strategy == 'cls':
            return x[:, 0]
        mask = mask.unsqueeze(-1).expand_as(x)
        if self.strategy == 'mean':
            return torch.sum(x * mask, dim=1) / torch.sum(mask, dim=1)
        elif self.strategy == 'max':
            return torch.max(x + (1 - mask) * -1e9, dim=1)[0]

class MedicalEmbedder(nn.Module):
    """
    A Transformer-based text embedder.
    Takes token IDs as input and returns a single embedding vector for each sequence.
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=3, d_ff=1024, pooling_strategy='cls', dropout=0.1):
        super(MedicalEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.pooling = Pooling(strategy=pooling_strategy)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens):
        """
        Args:
            src_tokens: A tensor of token IDs with shape (batch_size, sequence_length).
        Returns:
            A tensor of embedding vectors with shape (batch_size, d_model).
        """
        src_mask_4d = (src_tokens != 0).unsqueeze(1).unsqueeze(2)
        x = self.embedding(src_tokens)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask_4d)
            
        pooling_mask = (src_tokens != 0)
        return self.pooling(x, pooling_mask)