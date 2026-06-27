"""Transformer backbone used inside JEPA.

The MLP block maps per-atom token features into the embedding space consumed
by the JEPA frame for context/target comparison.
"""

import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        """Build a feed-forward block used inside the JEPA transformer."""
        assert n_layers >= 2
        super(MLP, self).__init__()
        self.map = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(0, n_layers-2):
            self.map.append(nn.Linear(hidden_dim, hidden_dim))
        self.map.append(nn.Linear(hidden_dim, out_dim))
        self.act = nn.SiLU()
        self.n_layers = n_layers
    
    def forward(self, x):
        """Apply the MLP layers with SiLU activations between hidden blocks."""
        for i in range(self.n_layers-1):
            x = self.act(self.map[i](x))
        x = self.map[-1](x)
        return x

class MHA(nn.Module):
    def __init__(self, attn_head, dim, dropout):
        """Create a masked multi-head self-attention layer for JEPA."""
        super(MHA, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dim = dim
        self.attn_head = attn_head
        self.softmax = nn.Softmax(dim=-1)
        self.WO = nn.Linear(dim, dim)
    
    def forward(self, q, k, v, mask):
        """Compute masked self-attention over the crystal token sequence."""
        b, lq, _ = q.shape
        b, kq, _ = k.shape
        b, vq, _ = v.shape

        dim_head = self.dim // self.attn_head
        q = q.reshape(b, lq, self.attn_head, dim_head).transpose(1, 2)
        k = k.reshape(b, kq, self.attn_head, dim_head).transpose(1, 2)
        v = v.reshape(b, vq, self.attn_head, dim_head).transpose(1, 2)
        
        mask = mask.unsqueeze(1).unsqueeze(-1).float() # (b, 1, l, 1)
        mask = mask @ mask.transpose(-1, -2) # (b, 1, l, l)

        attn_scores = q @ k.transpose(2, 3) / math.sqrt(dim_head) # (b, h, l, l)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-1e3'))
        attn = self.softmax(attn_scores) # (b, h, l, l)
        attn = self.dropout(attn) # (b, h, l, l)
        
        attn_out = (attn @ v).transpose(1, 2).reshape(b, lq, self.dim) # (b, l, d)
        attn_out = self.dropout(attn_out) # (b, l, d)
        attn_out = self.WO(attn_out) # (b, l, d)
        return attn_out

class Decoder_layer(nn.Module):
    def __init__(self, dim, attn_head, dropout):
        """Combine masked self-attention with a residual MLP block."""
        assert dim % attn_head == 0 
        super(Decoder_layer, self).__init__()
        self.LN_MHA_SA = nn.LayerNorm(dim) # SA for self-attention
        self.LN_MLP = nn.LayerNorm(dim) # MLP for feed-forward block
        
        self.qkv = nn.Linear(dim, 3*dim) # linear projection for query, key, value
        self.mha_sa = MHA(attn_head, dim, dropout) # masked multi-head self-attention
        self.MLP = nn.Sequential(nn.Linear(dim, 4*dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
    
    def forward(self, h, mask):
        """Run one transformer decoder block over the padded batch."""
        ## self_attention
        h[~mask] = 0. # zero out the padded tokens
        h_ini = h # save the input for residual connection
        h = self.LN_MHA_SA(h)  # apply layer normalization before self-attention
        q, k, v = self.qkv(h).chunk(3, dim=-1) # split into query, key, value
        sa_attn_out = self.mha_sa(q, k, v, mask) # compute masked self-attention output
        sa_attn_out[~mask] = 0. # zero out the padded tokens in the attention output
        h = h_ini + sa_attn_out # add the residual connection

        ## add & norm
        h = h + self.MLP(self.LN_MLP(h)) # apply MLP block to h
        h[~mask] = 0. # zero out the padded tokens after MLP
        return h
    
class Transformer(nn.Module):
    def __init__(self, hidden_dim, layers, attn_head, dropout):
        """Build the JEPA transformer encoder used for crystal embeddings."""
        super(Transformer, self).__init__()

        self.pe_emb = nn.Parameter(torch.zeros(500, hidden_dim)) # positional embedding for up to 500 tokens
        nn.init.trunc_normal_(self.pe_emb, std=0.02) # initialize positional embedding with truncated normal distribution

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) # class token for the transformer initialised with zeros
        nn.init.trunc_normal_(self.cls_token, std=0.02) # initialize class token with truncated normal distribution for stability

        self.block = nn.ModuleList() # create a list of decoder layers for the transformer
        for _ in range(layers):
            self.block.append(Decoder_layer(hidden_dim, attn_head, dropout))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, mask):
        """Encode a padded crystal batch into contextual embeddings."""
        b, l = mask.shape # get the batch size and sequence length from the mask

        cls_tokens = self.cls_token.expand(b, 1, -1) # expand the class token to match the batch size
        h = torch.cat([cls_tokens, h], 1) # concatenate the class token to the input sequence
        p_emb = self.pe_emb[:l].unsqueeze(0).repeat(b, 1, 1) # get the positional embeddings for the input sequence and repeat for the batch size
        h = h + p_emb # add the positional embeddings to the input sequence

        for i in range(len(self.block)): 
            h[~mask] = 0.
            h = self.block[i](h, mask)
        h = self.norm(h)
        h[~mask] = 0.
        return h