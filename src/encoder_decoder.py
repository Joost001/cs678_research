import random
import math
import copy


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Transformer

import ipdb

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0):
        super(MultiHeadedAttention, self).__init__()


        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # (30, 20, 80)
        # 

        assert (
            embed_size % heads == 0
        ), "Embedding size needs to be divisible by heads"

        self.linears = [nn.Linear(embed_size, embed_size) for _ in range(3)]
        self.out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        # ipdb.set_trace()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # (3, 7, 8) (3, 8, 7)
        # (3, 1, 7, 1)x
        # ipdb.set_trace()
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) 
        
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn

    def forward(self, queries, keys, values , mask=None):

        # Get number of training examples
        n_batch = queries.size()[0]

        
        
        # Pass the query, the key and the value through linear layers
        # We split the embedding dimenssion in multiple heads
        # Then swap dim 1 and dim 2
        # i.e dim values = (1, 80, 3000). (batch_size, n_words or frames, embedding)
        # With the split, we have the embedding dim to have 10 heads
        # We now have dim values = (1, 80, 10, 300)
        # then we need to swap the dimension
        # so we have this in the end (1, 10, 80, 300) 
        queries, keys, values = \
            [linear(x).view(n_batch, -1, self.heads, self.head_dim).transpose(1, 2)
             for linear, x in zip(self.linears, (queries, keys, values))]

        # ipdb.set_trace() 

        # ipdb.set_trace()
        # ipdb.set_trace()

        # mask = mask.view(n_batch, -1, self.heads, self.head_dim).transpose(1, 2)

        Z, attn = self.attention(query=queries, key=keys, value=values, mask=mask, 
                                 dropout=self.dropout)

        

        Z = Z.transpose(1, 2).contiguous() \
             .view(n_batch, -1, self.heads * self.head_dim)

        out = self.out(Z)

        del queries
        del keys
        del values

        return out, attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN"

    def __init__(self, embed_size, inner_layer_size, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.layer1 = nn.Linear(embed_size, inner_layer_size)
        self.layer2 = nn.Linear(inner_layer_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.layer1(X)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.layer2(out)
       

        return out

class EncoderLayer(nn.Module):

    def __init__(self, embed_size, heads, inner_layer_size):
        super(EncoderLayer, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.inner_layer_size = inner_layer_size

        self.muti_head_attn = MultiHeadedAttention(self.embed_size, self.heads)
        self.ffn = PositionwiseFeedForward(self.embed_size, self.inner_layer_size)

        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

    def forward(self, X, mask=None):
       
        # ipdb.set_trace() 

        out, attn =  self.muti_head_attn(X, X, X, mask)
        # Add skip connection
        out += X
        norm1_out = self.norm1(out)

        out = self.ffn(norm1_out)

        # Add skip connection
        out += norm1_out
        out = self.norm2(out)

        return out, attn

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, embed_size, heads, inner_layer_size, n_blocks):
        super(Encoder, self).__init__()
        
        ## Need to have the input from Jonathan
        ## Need to have the positional embedding

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_size,
                    heads,
                   inner_layer_size
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, X, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            X, attn = layer(X, mask)
        return X


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, embed_size, heads, inner_layer_size):
        super(DecoderLayer, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.inner_layer_size = inner_layer_size
        
        self.masked_muti_head_attn = MultiHeadedAttention(self.embed_size, self.heads)
        self.muti_head_attn = MultiHeadedAttention(self.embed_size, self.heads)
        self.ffn = PositionwiseFeedForward(self.embed_size, self.inner_layer_size)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)
        self.norm3 = nn.LayerNorm(self.embed_size)

        # self.linear = nn.Linear()
        

    def forward(self, X, memory, src_mask=None, tgt_mask=None):
        "Follow Figure 1 (right) for connections."

        # ipdb.set_trace()
        

        # query, key, value
        masked_m_h_out, _ =  self.masked_muti_head_attn(X, X, X, tgt_mask)
        masked_m_h_out = self.norm1(masked_m_h_out)
        masked_m_h_out += X

        # ipdb.set_trace()

        m_h_out, _ = self.muti_head_attn(masked_m_h_out, memory, memory, src_mask)
        m_h_out = self.norm2(m_h_out)
        m_h_out = m_h_out + masked_m_h_out
        
        out = self.ffn(m_h_out)
        # out += self.norm3(out)

        # ipdb.set_trace()
        
        return out, masked_m_h_out, m_h_out

class Decoder(nn.Module):
    "Core Decoder is a stack of N layers"

    def __init__(self, embed_size, heads, inner_layer_size, n_blocks, vocab_size):
        super(Decoder, self).__init__()
        
        ## Need to have the input from Jonathan
        ## Need to have the positional embedding

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    embed_size,
                    heads,
                   inner_layer_size
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, X,  memory, src_mask=None, tgt_mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            X, _, _ = layer(X, memory, src_mask, tgt_mask)
        
        return X


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float=0,
                 maxlen: int = 5000):
                 
        super(PositionalEncoding, self).__init__()

        # ipdb.set_trace()

        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
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
    def __init__(self, 
                embed_size: int, 
                heads: int, 
                inner_layer_size: int, 
                n_blocks: int, 
                vocab_size: int,
                dropout: int = 0):
        
        super(CustomTransformer, self).__init__()


        self.encoder = Encoder(embed_size, heads, inner_layer_size, n_blocks)
        self.decoder = Decoder(embed_size, heads, inner_layer_size, n_blocks, vocab_size)
        self.out = nn.Linear(embed_size, vocab_size)

        self.tok_emb = TokenEmbedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(emb_size=embed_size, dropout=dropout)


    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):


        # ipdb.set_trace() 

        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(self.tok_emb(tgt))
        
        out = self.encoder(src_emb, src_padding_mask)

        # (3, 1, 7, 1) 
        # (3, 4, 7, 7)
        # query key => target, key=> encoder
        # Decoder = query (3, 4, 10, 2) key (3, 4, 7, 2) => (3, 4, 10, 7) => (3, 1, 10, 1 ) => (3, 1, 10, 10 )

        # ipdb.set_trace()
        out = self.decoder(tgt_emb, out, tgt_padding_mask, tgt_mask)

        out = self.out(out)

        return out



def create_mask(src, tgt, pad_idx):

   
    n_samples, tgt_len = tgt.shape

  
    src_padding_mask = (src != pad_idx)[:, :, :1].unsqueeze(1)
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    
    tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
        n_samples, 1, tgt_len, tgt_len
    )

    return src_padding_mask, tgt_padding_mask, tgt_mask


if __name__ == "__main__":
    embed_size = 8
    heads = 4
    inner_layer_size = 32
    vocab_size = 500
    n_blocks = 5
    padding_indx = 0

    batch_size = 3
    n_epochs = 2

    N = 100
    all_data_size = batch_size * N

    # X_t = torch.rand(80, batch_size, 3000)
    # Y_t = torch.rand(12, batch_size, 3000)

    X = torch.rand(all_data_size, 7, embed_size)

    X[0, 5:, :] = padding_indx
    X[2, 4:, :] = padding_indx
    Y = torch.randint(2, vocab_size, (all_data_size, 10))
    Y[0, 7:] = padding_indx
    Y[1, 9:] = padding_indx
    Y[2, 5:] = padding_indx



    

    model = CustomTransformer(embed_size=embed_size, 
                                heads=heads,
                                inner_layer_size=inner_layer_size, 
                                n_blocks=n_blocks, 
                                vocab_size=vocab_size)


    # result = torch.argmax(F.softmax(logits, dim=-1), dim=-1)


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_indx)


    model.train()

    for epoch in range(30):
        losses = 0

        for i in range(N):

            optimizer.zero_grad()
            src = X[i*batch_size: (i+1) * batch_size, :, :]
            tgt = Y[i*batch_size:(i+1) * batch_size, :]
            
            tgt_input = tgt[:, :-1]

            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, tgt_input, padding_indx)  

            logits = model(src, tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
            
            
            # ipdb.set_trace()
            

            tgt_out = tgt[:, 1:]

            # ipdb.set_trace()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()    

            # ipdb.set_trace()

            # print(f"Finished epoch {i}, loss {loss.item()}")

        print(f"Epoch {epoch}, loss {losses/X.shape[0]}")

    ipdb.set_trace()

# salloc--partition=gpuq--qos=gpu–n 4 --ntasks-per-node=12 --gres=gpu:A100.40gb:4
# salloc -p gpuq -q gpu --ntasks-per-node=4 --gres=gpu:A100.40gb:1 -t 0-01:00:00 