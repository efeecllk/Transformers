import torch
import torch.nn as nn
# Self Attention Layer

class SelfAttention(nn.Module):
    def __init__(self,embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimensions = embed_size
        assert (self.heads * self.head_dimensions == embed_size), "Embed size must be divisible by heads"

        self.values = nn.Linear(self.head_dimensions, self.head_dimensions, bias=False)
        self.keys = nn.Linear(self.head_dimensions, self.head_dimensions, bias=False)
        self.queries = nn.Linear(self.head_dimensions, self.head_dimensions, bias=False)
        self.fullyConnected_out = nn.Linear(heads*self.head_dimensions, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # Split embedding into self.heads pieces


