import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (embed_size % heads == 0), "Embedding size needs to be divisible by heads"

        self.keys_linear = nn.Linear(embed_size, embed_size)
        self.queries_linear = nn.Linear(embed_size, embed_size)
        self.values_linear = nn.Linear(embed_size, embed_size)
        
        self.out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        Compute 'Scaled Dot Product Attention'
        :param query:
        :param key:
        :param value:
        :param mask:
        :param dropout:
        :return:
        """
        # ipdb.set_trace()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ipdb.set_trace()

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) 
        
        p_attn = F.softmax(scores, dim=-1)
        
        # why this?
        # if dropout is not None:
        #     p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn

    def forward(self, queries, keys, values, mask=None):
        """
        # Pass the query, the key and the value through linear layers
        # We split the embedding dimension in multiple heads
        # Then swap dim 1 and dim 2
        # i.e. dim values = (1, 80, 3000). (batch_size, n_words or frames, embedding)
        # With the split, we have the embedding dim to have 10 heads
        # We now have dim values = (1, 80, 10, 300)
        # then we need to swap the dimension
        # so we have this in the end (1, 10, 80, 300)
        :param queries:
        :param keys:
        :param values:
        :param mask:
        :return:
        """
        # Get number of training examples
        n_batch = queries.size()[0]

        queries = self.queries_linear(queries).view(n_batch, -1, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys_linear(keys).view(n_batch, -1, self.heads, self.head_dim).transpose(1, 2)
        values = self.values_linear(values).view(n_batch, -1, self.heads, self.head_dim).transpose(1, 2)

        # ipdb.set_trace()

        z, attn = self.attention(query=queries, key=keys, value=values, mask=mask,
                                 dropout=self.dropout)

        z = z.transpose(1, 2).contiguous() \
             .view(n_batch, -1, self.heads * self.head_dim)

        out = self.out(z)

        del queries
        del keys
        del values
        return out, attn


class PositionwiseFeedForward(nn.Module):
    """
    Implements Feed-Forward Network
    """
    def __init__(self, embed_size, inner_layer_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer1 = nn.Linear(embed_size, inner_layer_size)
        self.layer2 = nn.Linear(inner_layer_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.layer2(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, inner_layer_size, dropout = 0.1):
        super(EncoderLayer, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.inner_layer_size = inner_layer_size

        self.positional_encoding = PositionalEmbedding(emb_size=embed_size, dropout=dropout)
        self.multi_head_attn = MultiHeadedAttention(self.embed_size, self.heads)
        self.ffn = PositionwiseFeedForward(self.embed_size, self.inner_layer_size)

        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ipdb.set_trace() 
        out, attn = self.multi_head_attn(x, x, x, mask)
        out = self.dropout1(out)
        norm1_out = self.norm1(out+x)

        out = self.ffn(norm1_out)
        out = self.dropout2(out)
        out = self.norm2(out+norm1_out)
        
        return out, attn


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """
    def __init__(self, embed_size, heads, inner_layer_size, n_blocks):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([EncoderLayer(embed_size, heads, inner_layer_size) for _ in range(n_blocks)])

    def forward(self, x, mask=None):
        """
        Pass the input (and mask) through each layer in turn
        :param x:
        :param mask:
        :return:
        """
        for layer in self.layers:
            x, attn = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """
    def __init__(self, embed_size, heads, inner_layer_size, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.inner_layer_size = inner_layer_size
        
        self.masked_multi_head_attn = MultiHeadedAttention(self.embed_size, self.heads)
        self.multi_head_attn = MultiHeadedAttention(self.embed_size, self.heads)
        self.ffn = PositionwiseFeedForward(self.embed_size, self.inner_layer_size)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)
        self.norm3 = nn.LayerNorm(self.embed_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # self.linear = nn.Linear()

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Follow Figure 1 (right) for connections.
        :param x:
        :param memory:
        :param src_mask:
        :param tgt_mask:
        :return:
        """

        # ipdb.set_trace()

        # query, key, value
        masked_m_h_out, _ = self.masked_multi_head_attn(x, x, x, tgt_mask)
        masked_m_h_out = self.norm1(masked_m_h_out+x)
        masked_m_h_out = self.dropout1(masked_m_h_out)
        

        # ipdb.set_trace()
        m_h_out, _ = self.multi_head_attn(masked_m_h_out, memory, memory, src_mask)
        m_h_out = self.dropout2(m_h_out)
        m_h_out = self.norm2(m_h_out+masked_m_h_out)
        
        out = self.ffn(m_h_out)
        out = self.dropout3(out)
        out = self.norm3(out+m_h_out)

        # ipdb.set_trace()
        
        return out, masked_m_h_out, m_h_out


class Decoder(nn.Module):
    """
    Core Decoder is a stack of N layers
    """
    def __init__(self, embed_size, heads, inner_layer_size, n_blocks):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([DecoderLayer(embed_size, heads, inner_layer_size) for _ in range(n_blocks)])
        

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Pass the input (and mask) through each layer in turn
        :param x:
        :param memory:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        for layer in self.layers:
            x, _, _ = layer(x, memory, src_mask, tgt_mask)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=5000):
        """
        class variables
        :param emb_size: (int): 
        :param dropout: (float):
        :param max_len: (int):
        """
        super(PositionalEmbedding, self).__init__()

        # ipdb.set_trace()

        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # ipdb.set_trace()
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class CustomTransformer(nn.Module):
    def __init__(self, embed_size, heads, inner_layer_size, n_blocks, vocab_size, dropout=0.2):
        """
        :param embed_size: (int):
        :param heads: (int):
        :param inner_layer_size: (int):
        :param n_blocks: (int):
        :param vocab_size: (int):
        :param dropout: (int):
        """
        super(CustomTransformer, self).__init__()
        self.encoder = Encoder(embed_size, heads, inner_layer_size, n_blocks)
        self.decoder = Decoder(embed_size, heads, inner_layer_size, n_blocks)
        self.generator = nn.Linear(embed_size, vocab_size)

        self.positional_encoding_enc = PositionalEmbedding(emb_size=embed_size, dropout=dropout)

        self.emb_dec = TokenEmbedding(vocab_size, embed_size)
        self.positional_encoding_dec = PositionalEmbedding(emb_size=embed_size, dropout=dropout)
        
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):
        # ipdb.set_trace() 
        src_emb = self.positional_encoding_enc(src)
        tgt_emb = self.positional_encoding_dec(self.emb_dec(tgt))
        
        # ipdb.set_trace()
        out = self.encoder(src_emb, src_padding_mask)
        out = self.decoder(tgt_emb, out, tgt_padding_mask, tgt_mask)

        out = self.generator(out)

        return out


def create_mask(src, tgt, pad_idx):
    src_padding_mask = (src != pad_idx)[:, :, :1].unsqueeze(1)
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    
    tgt_mask = make_tgt_mask(tgt)

    return src_padding_mask, tgt_padding_mask, tgt_mask

def make_tgt_mask(tgt):
    batch_size, tgt_len = tgt.shape
    mask = (torch.triu(torch.ones((tgt_len, tgt_len)))).expand(batch_size, 1, tgt_len, tgt_len)
    return mask