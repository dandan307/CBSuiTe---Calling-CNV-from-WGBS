import numpy as np
import torch
from performer_pytorch import Performer
from einops import rearrange, repeat
import torch.nn as nn
import pandas as pd
import os
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEmbedding(nn.Module):
    
        def __init__(self, channels):
            super(PositionalEmbedding, self).__init__()
            inv_freq = 1. / (1000000000 ** (torch.arange(0, channels, 2).float() / channels))
            self.register_buffer('inv_freq', inv_freq)
            
            
        def forward(self, tensor,strt,ends):
            siz = 2001
            bs = tensor.shape[0]
           
            pos = torch.linspace(strt[0,0].item(), ends[0,0].item(), siz, device=tensor.device).type(self.inv_freq.type()) 

            sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)

            emb = emb[None,:,:]
            
            for i in range(1,bs):
                pos = torch.linspace(strt[i,0].item(), ends[i,0].item(), siz, device=tensor.device).type(self.inv_freq.type()) 
                sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
                emb_i = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
                emb_i = emb_i[None,:,:]
                emb = torch.cat((emb, emb_i), 0)
            return emb

class ChromosomeEmbedding(nn.Module):
    def __init__(self, channels, bin_size):
        super(ChromosomeEmbedding, self).__init__()
        self.dim = channels
        self.ce = nn.Parameter(torch.randn(24, self.dim))
        self.bin_size = bin_size
        
    def forward(self, tensor, chr):
        bs = tensor.shape[0]
        chr_mat_all = torch.zeros(bs, self.bin_size+1, self.dim).to(device)
        for i in range(bs):
            chr_mat = torch.unsqueeze(self.ce[int(chr[i].item()-1),:],dim=0).repeat_interleave(repeats=self.bin_size+1, dim = 0).to(device)
            chr_mat_all[i,:,:] = chr_mat

        return chr_mat_all

    
class CNVcaller(nn.Module):
    def __init__(self, bin_size, patch_size, depth, embed_dim, num_class, channels = 1):
        super().__init__()
        assert bin_size % patch_size == 0
        num_patches = (bin_size // patch_size) 
        patch_dim = channels * patch_size 
        self.patch_size = patch_size
        self.bin_size = bin_size
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.chr_emb = ChromosomeEmbedding(embed_dim, bin_size*2)
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)

        self.chromosome_token = nn.Parameter(torch.randn(1, 24, embed_dim))
        self.to_cls_token = nn.Identity()
        self.avgpooling = nn.Identity()
        self.attention = Performer(
            dim = embed_dim,
            depth = depth,
            heads = 8,
        )
              
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.mlp_head2 = nn.Linear(embed_dim, num_class)


    def forward(self, data, mask):
        chrs = data[:,:,-1]
        strt = data[:,:,-2]
        ends = data[:,:,-3]

        p = self.patch_size

        all_ind = list(range(self.bin_size*2 + 1))
        
        indices = torch.tensor(all_ind).to(device)
        data = torch.index_select(data, 2, indices).to(device)
        
        x = rearrange(data, 'b c (h p1) -> b h (p1 c)', p1 = p)
        x = x.to(torch.float32)

        x = self.patch_to_embedding(x)

        x += self.pos_emb(x, strt, ends)
        
        x += self.chr_emb(x, chrs)
        
        x = self.attention(x,input_mask = mask)
       
        x = self.avgpooling(torch.mean(x, dim=1))

        x = self.mlp_head(x)

        x = self.mlp_head2(x)
        return x
