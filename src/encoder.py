import math

import torch
from torch import nn
import torch.nn.functional as f


def scaled_dot_product_attention(query, key, value, padding_mask=None):
    """
    Scaled dot product for the attention heads. From https://arxiv.org/pdf/1706.03762.pdf section 3.2.1.
    :param padding_mask:
    :param query:
    :param key:
    :param value:
    :return:
    """
    score = query.bmm(key.transpose(1, 2))
    score = score.Tensor.masked_fill_(padding_mask, -float('inf'))
    scale = math.sqrt(query.size(-1))
    softmax = f.softmax(score / scale, dim=-1)
    return softmax.bmm(value)


def position_encoding(seq_len, dim_model, device=torch.device("cpu")):
    """
    :param seq_len: int
    :param dim_model: int
    :param device: torch.device
    :return: Tensor
    """
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query, key, value):
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_q, dim_k):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query, key, value):
        return self.linear(torch.cat([h(query, key, value) for h in self.heads], dim=-1))


def feed_forward(dim_input=512, dim_feedforward=2048):
    """
    position-wise feed-forward network. defaults are from the paper
    :param dim_input:
    :param dim_feedforward:
    :return: nn.Module
    """
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input))


class Residual(nn.Module):
    def __init__(self, sublayer, dimension, dropout=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors):
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=6, dim_feedforward=2048, dropout=0.1):
        """
        Defaults are from the paper.
        :param dim_model:
        :param num_heads:
        :param dim_feedforward:
        :param dropout:
        """
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(MultiHeadAttention(num_heads, dim_model, dim_q, dim_k), dimension=dim_model,
                                  dropout=dropout)
        self.feed_forward = Residual(feed_forward(dim_model, dim_feedforward), dimension=dim_model, dropout=dropout)

    def forward(self, source):
        source = self.attention(source, source, source)
        return self.feed_forward(source)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, dim_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                                     for _ in range(num_layers)])

    def forward(self, source):
        seq_len, dimension = source.size(1), source.size(2)
        source += position_encoding(seq_len, dimension)
        for layer in self.layers:
            source = layer(source)
        return source
